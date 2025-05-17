import pygame

def load_map(filename):
    with open(filename, 'r') as f:
        return [list(map(int, line.strip().split())) for line in f if line.strip()]

def draw_map(screen, game_map, cell_size=40):
    for y, row in enumerate(game_map):
        for x, cell in enumerate(row):
            color = (0, 0, 0) if cell == 1 else (255, 255, 255)
            rect = pygame.Rect(x * cell_size, y * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (200, 200, 200), rect, 1)  # grid lines

def save_maps_as_images(map_files, cell_size=40):
    for idx, map_file in enumerate(map_files):
        game_map = load_map(map_file)
        rows, cols = len(game_map), len(game_map[0])
        surface = pygame.Surface((cols * cell_size, rows * cell_size))
        draw_map(surface, game_map, cell_size)
        pygame.image.save(surface, f"map{idx+1}.png")

def main():
    pygame.init()
    map_files = [f"map{i}.txt" for i in range(1, 6)]
    save_maps_as_images(map_files)
    current_map = 0
    game_map = load_map(map_files[current_map])
    rows, cols = len(game_map), len(game_map[0])
    cell_size = 40
    screen = pygame.display.set_mode((cols * cell_size, rows * cell_size))
    pygame.display.set_caption("Map Viewer")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    current_map = (current_map + 1) % len(map_files)
                    game_map = load_map(map_files[current_map])
                elif event.key == pygame.K_LEFT:
                    current_map = (current_map - 1) % len(map_files)
                    game_map = load_map(map_files[current_map])
                rows, cols = len(game_map), len(game_map[0])
                screen = pygame.display.set_mode((cols * cell_size, rows * cell_size))

        screen.fill((255, 255, 255))
        draw_map(screen, game_map, cell_size)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()