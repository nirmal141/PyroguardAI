import pygame, sys

pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Pygame Test Window")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((0, 128, 255))
    pygame.display.flip()

pygame.quit()
sys.exit()