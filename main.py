import sys
import pygame
import numpy as np
from math import hypot

pygame.init()

# ---------------------------------
# Display
# ---------------------------------
WIDTH, HEIGHT = 1900, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flux-Core MIG Simulator v2.1")
clock = pygame.time.Clock()

# ---------------------------------
# Simulation resolution
# ---------------------------------
SIM_W, SIM_H = 320, 180
SCALE_X = WIDTH / SIM_W
SCALE_Y = HEIGHT / SIM_H

# ---------------------------------
# Colors
# ---------------------------------
BG = (18, 18, 18)
WHITE = (255, 255, 255)
ORANGE = (255, 140, 0)
YELLOW = (255, 220, 80)

# ---------------------------------
# Machine settings
# ---------------------------------
amps = 140
voltage = 18.0

# ---------------------------------
# Weld model
# ---------------------------------
class WeldPlate:
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.ambient = 20.0
        self.melt_temp = 900.0
        self.diffusion = 0.12
        self.cool_base = 0.05
        self.cool_weld = 0.09
        self._init_plate()

    def _init_plate(self):
        self.temp = np.full((self.h, self.w), self.ambient, dtype=np.float32)
        self.height = np.zeros((self.h, self.w), dtype=np.float32)
        self.material = np.zeros((self.h, self.w), dtype=np.uint8)
        gap = 4
        mid = self.w // 2
        self.material[:, :mid - gap] = 1
        self.material[:, mid + gap:] = 1
        self.height[self.material == 1] = 1.0

    def reset(self):
        self._init_plate()

    # ---------------------------------
    # Heat application
    # ---------------------------------
    def apply_heat(self, gx, gy, radius, power):
        r = int(radius)
        x0 = max(0, gx - r)
        x1 = min(self.w, gx + r + 1)
        y0 = max(0, gy - r)
        y1 = min(self.h, gy + r + 1)

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dx = xx - gx
        dy = yy - gy
        dist = np.sqrt(dx * dx + dy * dy)
        mask = (dist <= r)
        metal = self.material[y0:y1, x0:x1] != 0
        falloff = (1.0 - dist / r) * mask * metal
        # smooth deposition
        self.temp[y0:y1, x0:x1] += power * falloff * 0.25

    # ---------------------------------
    # Physics step
    # ---------------------------------
    def step(self):
        t = self.temp

        # Diffusion (clamped edges)
        up    = np.vstack((t[1:], t[-1:]))
        down  = np.vstack((t[:1], t[:-1]))
        left  = np.hstack((t[:,1:], t[:,-1:]))
        right = np.hstack((t[:,:1], t[:,:-1]))
        t += ((up + down + left + right) * 0.25 - t) * self.diffusion

        # Cooling
        base = self.material == 1
        weld = self.material == 2
        t[base] -= (t[base] - self.ambient) * self.cool_base
        t[weld] -= (t[weld] - self.ambient) * self.cool_weld
        t[t < self.ambient] = self.ambient

        # Melting / deposition
        molten = t > self.melt_temp
        self.material[molten] = 2

        # Filler deposition (bridging)
        hot = t > (self.melt_temp * 0.85)
        air = self.material == 0
        molten_neighbors = (
            np.roll(molten, 1, 0) | np.roll(molten, -1, 0) |
            np.roll(molten, 1, 1) | np.roll(molten, -1, 1)
        )
        deposit = hot & air & molten_neighbors
        self.material[deposit] = 2
        self.height[deposit] += (t[deposit] - self.melt_temp * 0.79) * 0.006

        # Excess temperature deposition
        excess = np.clip(t - self.melt_temp, 0, 1200)
        self.height[molten] += excess[molten] * 0.0008

        # Burn-through
        burn = molten & (excess > 700)
        self.height[burn] *= 0.985

        # Gravity / smoothing
        h = self.height
        self.height = (
            0.994 * h +
            0.006 * (
                np.roll(h, 1, 0) + np.roll(h, -1, 0) +
                np.roll(h, 1, 1) + np.roll(h, -1, 1)
            ) * 0.25
        )

    # ---------------------------------
    # Normals for lighting
    # ---------------------------------
    def compute_normals(self):
        noise = (np.random.rand(*self.height.shape) - 0.5) * 0.02
        h = self.height + noise
        dx = np.roll(h, -1, 1) - np.roll(h, 1, 1)
        dy = np.roll(h, -1, 0) - np.roll(h, 1, 0)
        nz = np.ones_like(h) * 1.6
        nx = -dx
        ny = -dy
        length = np.sqrt(nx*nx + ny*ny + nz*nz)
        return nx/length, ny/length, nz/length

    # ---------------------------------
    # Rendering
    # ---------------------------------
    def render(self):
        rgb = np.zeros((self.h, self.w, 3), dtype=np.float32)
        base = self.material == 1
        weld = self.material == 2
        rgb[base] = (0.38, 0.40, 0.45)
        rgb[weld] = (0.30, 0.32, 0.36)

        nx, ny, nz = self.compute_normals()
        light_dir = np.array([0.4, -0.3, 0.85])
        light_dir /= np.linalg.norm(light_dir)
        diffuse = np.clip(nx*light_dir[0] + ny*light_dir[1] + nz*light_dir[2], 0, 1)
        view = np.array([0, 0, 1])
        half = (light_dir + view)
        half /= np.linalg.norm(half)
        heat = np.clip((self.temp - 500) / 900, 0, 1)
        rough = np.clip(1 - heat, 0.25, 1.0)
        spec = np.clip(nx*half[0] + ny*half[1] + nz*half[2], 0, 1)
        spec = spec ** (12 + 48 * (1 - rough))
        rgb *= (0.35 + diffuse[..., None] * 0.75)
        rgb += spec[..., None] * 0.18

        # Heat glow
        glow = np.zeros_like(rgb)
        glow[...,0] = heat ** 1.5
        glow[...,1] = heat ** 2.4 * 0.6
        glow[...,2] = heat ** 4.0 * 0.25
        rgb = np.clip(rgb + glow, 0, 1)
        return (rgb * 255).astype(np.uint8)

# ---------------------------------
# Helpers
# ---------------------------------
def screen_to_grid(mx, my):
    return int(mx / SCALE_X), int(my / SCALE_Y)

def draw_ui():
    font = pygame.font.SysFont("consolas", 18)
    lines = [
        "Flux-Core MIG — Stack Dimes & Burn-Through Possible",
        f"Amps: {amps} (Q/E)  Voltage: {voltage:.1f} (A/D)",
        "LMB weld | R reset | ESC quit",
        f"FPS: {clock.get_fps():.1f}"
    ]
    y = 10
    for l in lines:
        screen.blit(font.render(l, True, WHITE), (10, y))
        y += 20

def draw_torch(pos, on):
    x, y = pos
    pygame.draw.circle(screen, ORANGE if on else WHITE, (x, y), 10)
    if on:
        pygame.draw.line(screen, YELLOW, (x, y), (x, y + 22), 2)

# ---------------------------------
# Main
# ---------------------------------
plate = WeldPlate(SIM_W, SIM_H)
welding = False
last_gx = last_gy = None
travel_speed = 0.0
REF_SPEED = 1.2

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_ESCAPE:
                running = False
            elif e.key == pygame.K_q:
                amps = max(40, amps - 10)
            elif e.key == pygame.K_e:
                amps = min(300, amps + 10)
            elif e.key == pygame.K_a:
                voltage = max(8.0, voltage - 0.5)
            elif e.key == pygame.K_d:
                voltage = min(28.0, voltage + 0.5)
            elif e.key == pygame.K_r:
                plate.reset()
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            welding = True
            last_gx = last_gy = None
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            welding = False

    mx, my = pygame.mouse.get_pos()

    if welding:
        gx, gy = screen_to_grid(mx, my)
        if last_gx is not None:
            raw_dist = hypot(gx - last_gx, gy - last_gy)
        else:
            raw_dist = 0.0

        # Exponential smoothing for travel speed
        travel_speed = travel_speed * 0.85 + raw_dist * 0.15
        last_gx, last_gy = gx, gy

        # Power and radius
        base_power = amps * 9 + voltage * 7.0
        speed_ratio = travel_speed / REF_SPEED
        travel_factor = np.clip(1.0 - 0.35 * speed_ratio, 0.65, 1.25)
        power = base_power * travel_factor
        radius = int(6 + (voltage - 10) * 0.35)

        plate.apply_heat(gx, gy, radius, power)

    plate.step()

    screen.fill(BG)
    rgb = plate.render()
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    surf = pygame.transform.smoothscale(surf, (WIDTH, HEIGHT))
    screen.blit(surf, (0, 0))

    draw_torch((mx, my), welding)
    draw_ui()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
