import sys
import pygame
import numpy as np

pygame.init()

# ---------------------------------
# Display
# ---------------------------------
WIDTH, HEIGHT = 1900, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Welding Simulator v1")
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

        self.temp = np.zeros((h, w), dtype=np.float32)
        self.height = np.zeros((h, w), dtype=np.float32)
        self.material = np.zeros((h, w), dtype=np.uint8)
        # 0 = air, 1 = base metal, 2 = weld metal

        self.ambient = 20.0
        self.melt_temp = 900.0

        self.diffusion = 0.16
        self.cool_base = 0.06
        self.cool_weld = 0.11

        self._init_plates()

    def _init_plates(self):
        gap = 4
        mid = self.w // 2
        self.material[:, :mid - gap] = 1
        self.material[:, mid + gap:] = 1
        self.height[self.material == 1] = 1.0

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
        mask = dist <= r
        falloff = (1.0 - dist / r) * mask

        self.temp[y0:y1, x0:x1] += power * falloff

    def step(self):
        t = self.temp

        # Diffusion
        up = np.roll(t, -1, 0)
        down = np.roll(t, 1, 0)
        left = np.roll(t, -1, 1)
        right = np.roll(t, 1, 1)
        t += ((up + down + left + right) * 0.25 - t) * self.diffusion

        # Exponential cooling
        base = self.material == 1
        weld = self.material == 2
        t[base] -= (t[base] - self.ambient) * self.cool_base
        t[weld] -= (t[weld] - self.ambient) * self.cool_weld
        t[t < self.ambient] = self.ambient

        # Melting / deposition
        molten = t > self.melt_temp
        self.material[molten] = 2
        self.height[molten] += (t[molten] - self.melt_temp) * 0.0045
        self.height[molten] *= 0.996
        self.height = np.clip(self.height, 0.0, 3.0)

    # ---------------------------------
    # TRUE 3D NORMAL COMPUTATION
    # ---------------------------------
    def compute_normals(self):
        h = self.height

        dx = np.roll(h, -1, 1) - np.roll(h, 1, 1)
        dy = np.roll(h, -1, 0) - np.roll(h, 1, 0)

        # Scale Z so height feels like depth
        nz = np.ones_like(h) * 1.5

        nx = -dx
        ny = -dy

        length = np.sqrt(nx*nx + ny*ny + nz*nz)
        nx /= length
        ny /= length
        nz /= length

        return nx, ny, nz

    def render(self):
        rgb = np.zeros((self.h, self.w, 3), dtype=np.float32)

        # Base albedo
        base = self.material == 1
        weld = self.material == 2

        rgb[base] = (0.38, 0.40, 0.45)
        rgb[weld] = (0.30, 0.32, 0.36)

        # ---------------------------------
        # Lighting
        # ---------------------------------
        nx, ny, nz = self.compute_normals()

        # Light direction (angled like shop lighting)
        light_dir = np.array([0.4, -0.3, 0.85])
        light_dir /= np.linalg.norm(light_dir)

        # Lambertian diffuse
        diffuse = np.clip(
            nx * light_dir[0] +
            ny * light_dir[1] +
            nz * light_dir[2],
            0, 1
        )

        # Specular (subtle)
        view_dir = np.array([0, 0, 1.0])
        half = (light_dir + view_dir)
        half /= np.linalg.norm(half)

        spec = np.clip(
            nx * half[0] +
            ny * half[1] +
            nz * half[2],
            0, 1
        ) ** 32

        # Apply lighting
        rgb *= (0.35 + diffuse[..., None] * 0.75)
        rgb += spec[..., None] * 0.15

        # ---------------------------------
        # Heat glow (additive, NOT shading)
        # ---------------------------------
        heat = np.clip(
            (self.temp - self.ambient) /
            (self.melt_temp - self.ambient),
            0, 1
        )

        glow = np.zeros_like(rgb)
        glow[..., 0] = heat * 1.0
        glow[..., 1] = heat * 0.55
        glow[..., 2] = heat * 0.15

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
        "Goal: Stack Dimes — Not Bird-shit",
        f"Amps: {amps} (Q/E)",
        f"Voltage: {voltage:.1f} (A/D)",
        "Left click weld | ESC quit",
        f"FPS: {clock.get_fps():.1f}",
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
print("\r\nConsole is enabled until bugs/errors are non-present\r\n")
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
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            welding = True
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            welding = False

    mx, my = pygame.mouse.get_pos()

    if welding:
        gx, gy = screen_to_grid(mx, my)
        power = (amps * 0.9 + voltage * 8.0) * 0.55
        plate.apply_heat(gx, gy, radius=9, power=power)

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
