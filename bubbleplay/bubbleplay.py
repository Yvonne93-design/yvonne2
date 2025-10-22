"""
Touch the Weather — Pygame prototype (Python)

Features:
- Mouse movement controls particle brush whose color/behavior maps to temperature-like parameter.
- Keyboard: number keys 1-9 select month presets; 'f' attempts to fetch HKO snapshot (requires internet);
- Visual feedback: HUD with current speed, selected month, and fetched weather summary.

Run:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python main.py

"""
import sys
import math
import random
import time
import threading

try:
    import pygame
except Exception:
    print('pygame not installed. See requirements.txt')
    raise

import requests
from bs4 import BeautifulSoup
import colorsys

# --- Config
WIDTH, HEIGHT = 1200, 800
FPS = 60

# month presets (1..12) simplified
MONTHS = [
    {'temp':16,'humidity':70,'wind':2},
    {'temp':16,'humidity':70,'wind':2},
    {'temp':18,'humidity':65,'wind':2},
    {'temp':21,'humidity':70,'wind':2},
    {'temp':24,'humidity':75,'wind':2.5},
    {'temp':27,'humidity':80,'wind':3},
    {'temp':29,'humidity':75,'wind':3.5},
    {'temp':29,'humidity':75,'wind':3.5},
    {'temp':27,'humidity':80,'wind':3},
    {'temp':25,'humidity':85,'wind':2.5},
    {'temp':22,'humidity':85,'wind':2},
    {'temp':18,'humidity':80,'wind':2},
]


class Particle:
    def __init__(self, x, y, vx, vy, color, life=1.5, size=None):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.life = life
        self.birth = time.time()
        self.prev_x = x
        self.prev_y = y
        self.split_done = False
        # size can be provided to control appearance; otherwise choose a wider random range
        if size is None:
            self.size = max(1, int(random.uniform(2, 12)))
        else:
            self.size = max(1, int(size))

    def age(self):
        return time.time() - self.birth

    def alive(self):
        return self.age() < self.life

    def update(self, dt, wind, blended_temp=None):
        # wind is (wx, wy)
        # store previous pos for line-like rendering
        self.prev_x = self.x
        self.prev_y = self.y

        self.vx += wind[0] * dt * 0.1
        self.vy += wind[1] * dt * 0.1
        # slight drag
        self.vx *= 0.999
        self.vy *= 0.999
        self.x += self.vx * dt
        self.y += self.vy * dt

        # possible splitting behavior influenced by blended_temp
        children = []
        try:
            if blended_temp is not None and not self.split_done:
                # map temperature to split probability per second
                # base around 18-30C: more splitting when warmer
                t = blended_temp
                # normalize roughly 10..35 -> 0..1
                prob_scale = max(0.0, min(1.0, (t - 18.0) / (35.0 - 18.0)))
                split_prob_per_sec = 0.25 * prob_scale  # at most 0.25 splits/sec
                if random.random() < split_prob_per_sec * dt:
                    # spawn 1-2 smaller child particles with deviated velocity
                    n = random.choice([1,2])
                    for i in range(n):
                        ang = math.atan2(self.vy, self.vx) + random.uniform(-1.2, 1.2)
                        sp = math.hypot(self.vx, self.vy) * random.uniform(0.4, 0.9) + random.uniform(10,30)
                        vx = math.cos(ang) * sp
                        vy = math.sin(ang) * sp
                        # child color slightly shifted based on temp
                        r = min(255, max(0, int(self.color[0] + random.uniform(-20, 20))))
                        g = min(255, max(0, int(self.color[1] + random.uniform(-20, 20))))
                        b = min(255, max(0, int(self.color[2] + random.uniform(-20, 20))))
                        child_size = random.uniform(2, 12)
                        child = Particle(self.x, self.y, vx, vy, (r,g,b), life=self.life * random.uniform(0.5, 0.9), size=child_size)
                        children.append(child)
                    # mark as split so it doesn't repeatedly split
                    self.split_done = True
        except Exception:
            pass
        return children


def fetch_hko_snapshot(summary_holder):
    """Fetch a simple summary (temp/humidity) from HKO page and store in summary_holder dict."""
    try:
        url = 'https://www.hko.gov.hk/sc/wxinfo/currwx/fnd.htm'
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        # crude parse: look for text like '31.2°C 65%'
        txt = soup.get_text(separator=' ')
        # find first occurrence of pattern like 'xx.x°C'
        import re
        m = re.search(r"(\d{1,2}(?:\.\d)?)\s*°C", txt)
        hum = re.search(r"(\d{1,2})%", txt)
        if m:
            temp = float(m.group(1))
        else:
            temp = None
        if hum:
            humidity = int(hum.group(1))
        else:
            humidity = None
        summary_holder['temp'] = temp
        summary_holder['humidity'] = humidity
        summary_holder['raw'] = 'Fetched HKO snapshot.'
    except Exception as e:
        summary_holder['raw'] = f'Fetch error: {e}'


def temp_to_color(temp):
    """Map temperature (C) to RGB color.
    Cold -> blueish, Warm -> orange/red
    """
    if temp is None:
        # default neutral
        return (180, 180, 220)
    # clamp 0..35
    t = max(0, min(35, temp)) / 35.0
    # hue interpolation blue (200) -> red (10) in HSL-ish
    # we'll produce RGB by mixing
    r = int(200 * t + 50 * (1 - t))
    g = int(80 * t + 100 * (1 - t))
    b = int(50 * t + 200 * (1 - t))
    return (r, g, b)


def _clamp(v, a, b):
    return max(a, min(b, v))


def draw_shaded_sphere(surface, cx, cy, radius, color, alpha=200, light_dir=(-1, -1), detail=8):
    """Draw a simple shaded sphere using concentric circles and a specular highlight.
    light_dir: tuple indicating light vector direction; values around (-1,-1) mean top-left light.
    """
    # normalize light dir
    lx, ly = light_dir
    mag = math.hypot(lx, ly) or 1.0
    lx /= mag; ly /= mag

    # base color
    rcol, gcol, bcol = color
    # draw concentric circles (farther from center darker, simulate shading)
    for i in range(detail, 0, -1):
        t = i / detail
        rr = int(radius * t)
        # shading factor: brighter toward light-facing side
        shade = 0.6 + 0.4 * (0.5 + 0.5 * (1 - t))
        # compute slight light offset per ring
        ox = int(-lx * (1 - t) * radius * 0.25)
        oy = int(-ly * (1 - t) * radius * 0.25)
        a = int(alpha * (0.5 + 0.5 * t) * 0.6)
        col = (int(_clamp(rcol * shade, 0, 255)), int(_clamp(gcol * shade, 0, 255)), int(_clamp(bcol * shade, 0, 255)), a)
        surf = pygame.Surface((rr*2, rr*2), pygame.SRCALPHA)
        pygame.draw.circle(surf, col, (rr, rr), rr)
        surface.blit(surf, (cx-rr+ox, cy-rr+oy))

    # specular highlight (small bright spot)
    hx = int(cx + -lx * radius * 0.35)
    hy = int(cy + -ly * radius * 0.35)
    hr = max(1, int(radius * 0.18))
    spec = pygame.Surface((hr*2, hr*2), pygame.SRCALPHA)
    pygame.draw.circle(spec, (255,255,255,int(alpha*0.6)), (hr,hr), hr)
    surface.blit(spec, (hx-hr, hy-hr))


def draw_surface_deformations(surface, cx, cy, radius, base_color, alpha, seed, t):
    """Draw small random ellipses on the sphere surface to simulate deforming patterns.
    seed: particle-specific seed to make variations consistent per particle
    t: current time to animate deforming
    """
    random.seed(int(seed) % 100000)
    num = random.randint(1, 4)
    for i in range(num):
        phase = random.random() * math.tau
        ang = random.random() * math.tau + math.sin(t*0.6 + phase) * 0.6
        # position on sphere surface
        rx = cx + math.cos(ang) * radius * random.uniform(0.2, 0.8)
        ry = cy + math.sin(ang) * radius * random.uniform(0.2, 0.8)
        w = int(radius * random.uniform(0.12, 0.4))
        h = int(w * random.uniform(0.6, 1.2))
        # color slightly darker/lighter than base
        dr = int(random.uniform(-30, 30))
        dg = int(random.uniform(-30, 30))
        db = int(random.uniform(-30, 30))
        col = (int(_clamp(base_color[0]+dr,0,255)), int(_clamp(base_color[1]+dg,0,255)), int(_clamp(base_color[2]+db,0,255)), int(alpha*0.25))
        ell = pygame.Surface((w*2, h*2), pygame.SRCALPHA)
        pygame.draw.ellipse(ell, col, (0,0,w*2,h*2))
        # animate small local jitter
        jx = int(math.sin(t*1.5 + seed + i) * 3)
        jy = int(math.cos(t*1.2 + seed*0.7 + i) * 3)
        surface.blit(ell, (int(rx - w + jx), int(ry - h + jy)))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption('Touch the Weather — Pygame Prototype')
    clock = pygame.time.Clock()

    particles = []
    max_particles = 1200

    running = True
    last_mouse = pygame.mouse.get_pos()
    last_time = time.time()
    selected_month = 10
    summary = {'temp': None, 'humidity': None, 'raw': 'No fetch yet'}

    font = pygame.font.SysFont(None, 20)

    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_f:
                    # spawn a thread to fetch HKO
                    threading.Thread(target=fetch_hko_snapshot, args=(summary,), daemon=True).start()
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    num = event.key - pygame.K_0
                    if 1 <= num <= 9:
                        selected_month = num

        mx, my = pygame.mouse.get_pos()
        now = time.time()
        dx = mx - last_mouse[0]
        dy = my - last_mouse[1]
        dist = math.hypot(dx, dy)
        speed = dist / max(1e-6, now - last_time)

        # determine base parameters
        base = MONTHS[(selected_month-1) % 12]
        # blend in fetched temp if available
        temp = summary.get('temp') if summary.get('temp') is not None else base['temp']
        color = temp_to_color(temp)

        # spawn particles when moving
        if dist > 1:
            count = int(min(30, max(1, speed * 0.02)))
            for i in range(count):
                if len(particles) > max_particles: break
                # sometimes use a random direction to create more organic spread
                if random.random() < 0.25:
                    angle = random.uniform(0, math.tau)
                else:
                    angle = math.atan2(dy, dx) + random.uniform(-0.8, 0.8)
                sp = speed * 0.03 + random.uniform(-20, 40) + base['wind']*10
                vx = math.cos(angle) * sp
                vy = math.sin(angle) * sp
                # jitter color by humidity
                r = min(255, max(0, int(color[0] + random.uniform(-30,30))))
                g = min(255, max(0, int(color[1] + random.uniform(-30,30))))
                b = min(255, max(0, int(color[2] + random.uniform(-30,30))))
                life = random.uniform(0.6, 2.0)
                size = random.uniform(3, 14)
                particles.append(Particle(mx + random.uniform(-6,6), my + random.uniform(-6,6), vx, vy, (r,g,b), life, size=size))

        # blended temperature smoothing (exponential moving toward fetched temp)
        if 'blended_temp' not in locals():
            # initialize blended_temp to base temp on first frame
            blended_temp = base['temp']
        fetched_temp = summary.get('temp')
        if fetched_temp is not None:
            # small smoothing factor per frame
            blend_alpha = 0.02
            blended_temp = blended_temp + (fetched_temp - blended_temp) * blend_alpha
        else:
            # slowly relax back toward base
            blended_temp = blended_temp + (base['temp'] - blended_temp) * 0.005

        # blended humidity smoothing (same idea)
        if 'blended_humidity' not in locals():
            blended_humidity = base.get('humidity', 70)
        fetched_hum = summary.get('humidity')
        if fetched_hum is not None:
            blend_h = 0.02
            blended_humidity = blended_humidity + (fetched_hum - blended_humidity) * blend_h
        else:
            blended_humidity = blended_humidity + (base.get('humidity',70) - blended_humidity) * 0.005

        # compute sky gradient colors from blended temp & humidity
        def temp_humidity_to_sky(t, h):
            # map temperature and humidity to two RGB colors (top/bottom) while avoiding
            # green midtones by interpolating between curated RGB palettes.
            hum = max(0.0, min(100.0, h))
            tnorm = (t - 10.0) / (35.0 - 10.0)
            tnorm = max(0.0, min(1.0, tnorm))

            # curated palettes (avoid pure greens):
            cold_top = (18, 48, 120)    # deep blue
            cold_bottom = (110, 160, 220)
            warm_top = (255, 165, 90)   # warm peach/orange
            warm_bottom = (255, 220, 160)

            # interpolate between cold and warm by temperature
            def lerp(c1, c2, a):
                return (int(c1[0]*(1-a) + c2[0]*a), int(c1[1]*(1-a) + c2[1]*a), int(c1[2]*(1-a) + c2[2]*a))

            top = lerp(cold_top, warm_top, tnorm)
            bottom = lerp(cold_bottom, warm_bottom, tnorm)

            # humidity -> desaturate toward gray and lower contrast
            gray = (200,200,210)
            cloudiness = hum/100.0
            desat = min(0.7, cloudiness * 0.7)
            def desat_col(col):
                return (int(col[0]*(1-desat) + gray[0]*desat), int(col[1]*(1-desat) + gray[1]*desat), int(col[2]*(1-desat) + gray[2]*desat))

            top = desat_col(top)
            bottom = desat_col(bottom)

            # slightly darken bottom when humid to simulate lower contrast
            bottom = (int(bottom[0]* (1 - cloudiness*0.15)), int(bottom[1]* (1 - cloudiness*0.15)), int(bottom[2]* (1 - cloudiness*0.15)))

            return top, bottom

        def draw_cloud_layer(surface, humidity):
            """Draw multiple cached cloud layers with independent randomized shapes and
            parallax motion. Each layer caches its generated Surface and regenerates
            only when its own regen interval elapses or when humidity changes enough.
            """
            w, h = surface.get_size()
            # module-level cache for multiple layers
            global _cloud_layers
            try:
                _cloud_layers
            except NameError:
                _cloud_layers = {'layers': []}

            cloudiness = max(0.0, min(1.0, humidity / 100.0))

            # decide number of layers: always at least 2, more when very humid
            target_layers = 2 + int(cloudiness * 3)  # 2..5

            # ensure cache has the right number of layer slots
            while len(_cloud_layers['layers']) < target_layers:
                _cloud_layers['layers'].append({'surface': None, 'hum': None, 'ts': 0.0, 'seed': random.randint(0, 2**30), 'regen_interval': 2.5 + random.random() * 3.5, 'speed': random.uniform(-0.02, 0.02), 'yoff': random.uniform(0.05, 0.45), 'alpha_mul': random.uniform(0.4, 1.0)})
            # trim if too many
            if len(_cloud_layers['layers']) > target_layers:
                _cloud_layers['layers'] = _cloud_layers['layers'][:target_layers]

            now = time.time()

            for idx, layer in enumerate(_cloud_layers['layers']):
                regen = False
                # small per-layer thresholds so shapes vary independently
                HUM_THRESHOLD = 6.0
                if layer['surface'] is None:
                    regen = True
                elif abs((layer.get('hum') or 0) - humidity) > HUM_THRESHOLD:
                    regen = True
                elif now - (layer.get('ts') or 0) > layer.get('regen_interval', 4.0):
                    regen = True

                if regen:
                    rnd = random.Random(layer.get('seed', 0) ^ int(humidity*17) ^ idx)
                    cloud_strength = cloudiness
                    cloud_surf = pygame.Surface((w, h), pygame.SRCALPHA)

                    # Helper: generate low-res fractal value noise and upscale to produce
                    # amorphous cloud silhouettes (distinct from brush particles).
                    def generate_value_noise(nw, nh, octaves=3, lacunarity=2.0, gain=0.5, seed_val=0):
                        # create a base random grid and bilinearly interpolate for smoothness
                        rng = random.Random(seed_val)
                        # Start with a small grid of random values
                        grid_w = max(3, nw // 8)
                        grid_h = max(3, nh // 8)
                        grid = [[rng.random() for _ in range(grid_w+1)] for __ in range(grid_h+1)]

                        def sample(u, v):
                            # u in [0,1), v in [0,1)
                            x = u * grid_w
                            y = v * grid_h
                            x0 = int(math.floor(x))
                            y0 = int(math.floor(y))
                            xf = x - x0
                            yf = y - y0
                            x1 = min(x0+1, grid_w)
                            y1 = min(y0+1, grid_h)
                            a = grid[y0][x0]
                            b = grid[y0][x1]
                            c = grid[y1][x0]
                            d = grid[y1][x1]
                            # bilinear
                            return (a*(1-xf)*(1-yf) + b*xf*(1-yf) + c*(1-xf)*yf + d*xf*yf)

                        # fractal sum of interpolated grids with different scales
                        out = [[0.0 for _ in range(nw)] for __ in range(nh)]
                        amp = 1.0
                        freq = 1.0
                        for o in range(octaves):
                            # create a temporary grid for this octave
                            # use the same generator but offset seed slightly
                            rng2 = random.Random(seed_val + o*997)
                            g_w = max(2, int(grid_w * freq))
                            g_h = max(2, int(grid_h * freq))
                            g = [[rng2.random() for _ in range(g_w+1)] for __ in range(g_h+1)]

                            for j in range(nh):
                                v = j / max(1, nh-1)
                                for i2 in range(nw):
                                    u = i2 / max(1, nw-1)
                                    # sample this octave's grid (bilinear)
                                    x = u * g_w
                                    y = v * g_h
                                    x0 = int(math.floor(x)); y0 = int(math.floor(y))
                                    xf = x - x0; yf = y - y0
                                    x1 = min(x0+1, g_w); y1 = min(y0+1, g_h)
                                    a = g[y0][x0]; b = g[y0][x1]; c = g[y1][x0]; d = g[y1][x1]
                                    val = (a*(1-xf)*(1-yf) + b*xf*(1-yf) + c*(1-xf)*yf + d*xf*yf)
                                    out[j][i2] += val * amp
                            amp *= gain
                            freq *= lacunarity

                        # normalize
                        maxv = max(max(row) for row in out) or 1.0
                        minv = min(min(row) for row in out) or 0.0
                        rngv = maxv - minv if maxv - minv != 0 else 1.0
                        for j in range(nh):
                            for i2 in range(nw):
                                out[j][i2] = (out[j][i2] - minv) / rngv
                        return out

                    # generate a low-res noise map for silhouette then upscale
                    nw = max(160, w // (6 + idx))
                    nh = max(80, h // (10 + idx))
                    noise = generate_value_noise(nw, nh, octaves=3, seed_val=layer.get('seed', 0) ^ idx)

                    # paint the noise map into a small surface (color + alpha), then smoothscale
                    small = pygame.Surface((nw, nh), pygame.SRCALPHA)
                    max_alpha = int(140 * (0.6 + cloud_strength*0.8) * layer['alpha_mul'])
                    tint_r = int(230 + rnd.randint(-12, 12))
                    tint_g = int(230 + rnd.randint(-10, 10))
                    tint_b = int(230 + rnd.randint(-8, 14))
                    for j in range(nh):
                        for i2 in range(nw):
                            v = noise[j][i2]
                            # threshold to produce puff clusters; raise to power to shape
                            alpha = int((v ** (1.1 + rnd.random()*0.8)) * max_alpha)
                            if alpha <= 6:
                                # tiny chance to keep faint speck for texture
                                if rnd.random() < 0.02:
                                    alpha = 6
                                else:
                                    alpha = 0
                            small.set_at((i2, j), (tint_r, tint_g, tint_b, alpha))

                    scaled = pygame.transform.smoothscale(small, (w, h))
                    cloud_surf.blit(scaled, (0, 0))

                    # add a few larger soft ellipses for highlights within this layer (sparingly)
                    for i in range(max(1, int(1 + idx))):
                        cw = int(w * rnd.uniform(0.08, 0.22))
                        ch = int(cw * rnd.uniform(0.35, 0.6))
                        x = int(rnd.uniform(-cw*0.5, w - cw*0.5))
                        y = int(h * (layer['yoff'] + rnd.uniform(-0.02, 0.04)))
                        surf_blob = pygame.Surface((cw*2, ch*2), pygame.SRCALPHA)
                        a2 = int(60 * layer['alpha_mul'] * (0.8 + rnd.random()*0.6))
                        pygame.draw.ellipse(surf_blob, (250,250,250,a2), (0,0,cw*2,ch*2))
                        cloud_surf.blit(surf_blob, (x - cw, y - ch))

                    # sprinkle fine puffs for detail (but fewer than before)
                    small_n = int(6 + cloud_strength * 18)
                    for i in range(small_n):
                        cw = int(w * rnd.uniform(0.005, 0.03))
                        cw = max(3, cw)
                        ch = int(cw * rnd.uniform(0.6, 1.2))
                        x = rnd.randint(-cw, w)
                        y = int(h * (layer['yoff'] + rnd.uniform(-0.05, 0.08)))
                        surf_blob = pygame.Surface((cw*2, ch*2), pygame.SRCALPHA)
                        alpha = int(8 + rnd.uniform(0, 48) * layer['alpha_mul'] * cloud_strength)
                        pr = _clamp(230 + int(rnd.uniform(-12, 8)), 190, 255)
                        pg = _clamp(230 + int(rnd.uniform(-10, 6)), 190, 255)
                        pb = _clamp(230 + int(rnd.uniform(-6, 12)), 190, 255)
                        pygame.draw.ellipse(surf_blob, (pr, pg, pb, alpha), (0,0,cw*2,ch*2))
                        cloud_surf.blit(surf_blob, (x - cw//2 + rnd.randint(-6,6), y - ch//2 + rnd.randint(-4,4)))

                    # store into cache
                    layer['surface'] = cloud_surf
                    layer['hum'] = humidity
                    layer['ts'] = now
                    # refresh some layer params to vary subsequent regenerations
                    layer['seed'] = rnd.randint(0, 2**30)
                    layer['regen_interval'] = 2.0 + rnd.random() * 4.0
                    layer['speed'] = rnd.uniform(-0.03, 0.03)
                    layer['yoff'] = _clamp(layer['yoff'] + rnd.uniform(-0.02, 0.02), 0.02, 0.55)

                # blit the layer with slight horizontal parallax offset
                if layer.get('surface') is not None:
                    s = layer['surface']
                    # horizontal looping offset based on per-layer speed
                    speed = layer.get('speed', 0.01)
                    offset = int((now * speed * w) % w)
                    # tile the layer to allow continuous horizontal motion
                    surface.blit(s, (-offset, 0))
                    surface.blit(s, (w - offset, 0))

        def draw_vertical_gradient(surface, top_col, bottom_col):
            w,h = surface.get_size()
            grad = pygame.Surface((w,h))
            # use gamma to make the gradient non-linear (more pronounced)
            gamma = 1.6
            contrast = 0.25
            def apply_contrast(c):
                return int(max(0, min(255, (c-128) * (1+contrast) + 128)))
            for y in range(h):
                t = y / max(1, h-1)
                tlerp = pow(t, gamma)
                r = int(top_col[0] * (1-tlerp) + bottom_col[0] * tlerp)
                g = int(top_col[1] * (1-tlerp) + bottom_col[1] * tlerp)
                b = int(top_col[2] * (1-tlerp) + bottom_col[2] * tlerp)
                # slightly boost contrast to make gradient pop
                r = apply_contrast(r)
                g = apply_contrast(g)
                b = apply_contrast(b)
                pygame.draw.line(grad, (r,g,b), (0,y), (w,y))
            surface.blit(grad, (0,0))

        sky_top, sky_bottom = temp_humidity_to_sky(blended_temp, blended_humidity)

        # update particles; allow them to spawn children based on blended_temp
        wind = (math.cos(time.time()*0.2)*base['wind']*5, math.sin(time.time()*0.15)*base['wind']*3)
        new_children = []
        for p in particles:
            children = p.update(dt, wind, blended_temp=blended_temp)
            if children:
                new_children.extend(children)
        if new_children:
            particles.extend(new_children)


        particles = [p for p in particles if p.alive() and -100 < p.x < screen.get_width()+100 and -100 < p.y < screen.get_height()+100]

        # draw sky gradient background based on blended temp & humidity
        draw_vertical_gradient(screen, sky_top, sky_bottom)
        # cloud layer
        draw_cloud_layer(screen, blended_humidity)
        # overlay a slight translucent veil to keep motion trails visible
        veil = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        veil.fill((6,6,8,18))
        screen.blit(veil, (0,0))

        # draw particles with blended blob + subtle line to get fusion + streak
        for p in particles:
            age = p.age() / p.life
            base_alpha = max(0, int(200 * (1 - age)))

            # Draw a shaded 3D-like sphere for depth
            try:
                t_norm = (blended_temp - 10.0) / 25.0
            except Exception:
                t_norm = 0.5
            t_norm = max(0.0, min(1.0, t_norm))
            sphere_radius = int(max(6, p.size * (4 + t_norm * 10)))
            draw_shaded_sphere(screen, int(p.x), int(p.y), sphere_radius, p.color, alpha=base_alpha, light_dir=(-0.6, -0.4))

            # draw dynamic surface deformations on top
            draw_surface_deformations(screen, int(p.x), int(p.y), sphere_radius, p.color, base_alpha, seed=hash((int(p.x*7), int(p.y*13))), t=time.time())

            # subtle stretched line on top as streak (lower opacity)
            speed_mag = math.hypot(p.vx, p.vy)
            length = int(max(2, min(80, speed_mag * 0.5 + t_norm * 40)))
            width = max(1, int(p.size * (0.6 + t_norm * 0.8)))
            dx = p.x - p.prev_x
            dy = p.y - p.prev_y
            heading = math.atan2(dy, dx) if dx != 0 or dy != 0 else 0
            line_surf = pygame.Surface((length+4, width+4), pygame.SRCALPHA)
            col = (p.color[0], p.color[1], p.color[2], int(base_alpha * 0.28))
            pygame.draw.line(line_surf, col, (2, width//2+2), (length+2, width//2+2), width)
            rot = -math.degrees(heading)
            rsurf = pygame.transform.rotate(line_surf, rot)
            rrect = rsurf.get_rect(center=(int(p.x), int(p.y)))
            screen.blit(rsurf, rrect)

        # HUD
        hud_lines = [
            f'Speed: {int(speed)} px/s',
            f'Selected month (press 1-9): {selected_month}',
            f'Base temp: {base["temp"]} C • fetched temp: {summary.get("temp")}',
            f'Fetch status: {summary.get("raw")}'
        ]
        y = 8
        for ln in hud_lines:
            txt = font.render(ln, True, (240,240,240))
            screen.blit(txt, (8,y))
            y += 20

        pygame.display.flip()
        last_mouse = (mx,my)
        last_time = now

    pygame.quit()


if __name__ == '__main__':
    main()
