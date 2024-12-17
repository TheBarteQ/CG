# CG
Домашние работы и РК по компьютерной графике

### 1 Создать программу, которая рисует отрезок между двумя точками, заданными пользователем

```python
import matplotlib.pyplot as plt  # Импортируем как plt для вызова plt.show()
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np

# Функция для рисования отрезка между двумя точками
def draw_line(img, x0, y0, x1, y1, color):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        img.putpixel((x0, y0), color)
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

# Функция для рисования сетки
def draw_grid(img, step, color):
    width, height = img.size
    # Вертикальные линии
    for x in range(0, width, step):
        draw_line(img, x, 0, x, height-1, color)
    # Горизонтальные линии
    for y in range(0, height, step):
        draw_line(img, 0, y, width-1, y, color)

# Получаем координаты от пользователя
x0 = int(input("Input first x0-coordinate: "))
y0 = int(input("Input first y0-coordinate: "))
x1 = int(input("Input second x1-coordinate: "))
y1 = int(input("Input second y1-coordinate: "))

# Создаём пустое изображение
img = Image.new('RGB', (1000, 900), 'white')

# Рисуем сетку с шагом 50 пикселей 
draw_grid(img, 50, (200, 200, 200))  # Серая сетка

# Рисуем линию на основе введённых пользователем координат
draw_line(img, x0, y0, x1, y1, (0, 0, 0))

# Показываем изображение
imshow(np.asarray(img))

# Явно отображаем изображение
plt.show()

# Сохраняем изображение
img.save('Linia.png')
```

### 2 Создать программу, которая рисует окружность с заданным пользователем радиусом

```python
import matplotlib.pyplot as plt
import numpy as np

def bresenham_circle(radius):
    x = 0
    y = radius
    d = 3 - 2 * radius
    points = []

    def draw_circle_points(x, y):
        points.extend([
            (x, y), (-x, y), (x, -y), (-x, -y),
            (y, x), (-y, x), (y, -x), (-y, -x)
        ])

    while x <= y:
        draw_circle_points(x, y)
        if d <= 0:
            d = d + 4 * x + 6
        else:
            d = d + 4 * (x - y) + 10
            y -= 1
        x += 1

    return points

def plot_circle(radius):
    points = bresenham_circle(radius)
    
    # Удаляем дубликаты и сортируем точки по углам
    unique_points = list(set(points))
    unique_points.sort(key=lambda p: np.arctan2(p[1], p[0]))

    # Добавляем первую точку в конец для замыкания контура
    unique_points.append(unique_points[0])
    
    # Разворачиваем список точек в x и y для построения графика
    x_coords = [point[0] for point in unique_points]
    y_coords = [point[1] for point in unique_points]

    # Рисуем
    plt.plot(x_coords, y_coords, color='black')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'My MEGA GIPER ULTRA CIRCLE {radius}')
    plt.grid(True)
    plt.show()

#Enter
radius = int(input("R: "))
plot_circle(radius)
```

### 3 Циферблат

```python
import matplotlib.pyplot as plt
import numpy as np

def bresenham_circle(radius):
    x = 0
    y = radius
    d = 3 - 2 * radius
    points = []

    def draw_circle_points(x, y):
        points.extend([
            (x, y), (-x, y), (x, -y), (-x, -y),
            (y, x), (-y, x), (y, -x), (-y, -x)
        ])

    while x <= y:
        draw_circle_points(x, y)
        if d <= 0:
            d = d + 4 * x + 6
        else:
            d = d + 4 * (x - y) + 10
            y -= 1
        x += 1

    return points

def plot_circle_with_ticks(radius, num_ticks):
    points = bresenham_circle(radius)
    
    # Удаляем дубликаты и сортируем точки по углам
    unique_points = list(set(points))
    unique_points.sort(key=lambda p: np.arctan2(p[1], p[0]))

    # Добавляем первую точку в конец для замыкания контура
    unique_points.append(unique_points[0])
    
    # Разворачиваем список точек в x и y для построения графика
    x_coords = [point[0] for point in unique_points]
    y_coords = [point[1] for point in unique_points]

    fig, ax = plt.subplots()
    ax.plot(x_coords, y_coords, color='blue')

    # Добавляем засечки как на циферблате
    tick_length = 0.1 * radius
    for i in range(num_ticks):
        angle = 2 * np.pi * i / num_ticks
        x_tick_start = (radius - tick_length) * np.cos(angle)
        y_tick_start = (radius - tick_length) * np.sin(angle)
        x_tick_end = radius * np.cos(angle)
        y_tick_end = radius * np.sin(angle)
        
        # Рисуем засечки
        ax.plot([x_tick_start, x_tick_end], [y_tick_start, y_tick_end], color='red', lw=1.5)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Bresenham BASING {radius} and {num_ticks} ULTRAIMBA')
    ax.grid(True)
    plt.show()

radius = int(input("R: "))
num_ticks = 12
plot_circle_with_ticks(radius, num_ticks)

```

### 4 Реализация алгоритма Сезерленда-Коэна

```python
import matplotlib.pyplot as plt

# Opredelyaem kody regionov dlya otsecheniya
INSIDE = 0  # 0000
LEFT = 1    # 0001
RIGHT = 2   # 0010
BOTTOM = 4  # 0100
TOP = 8     # 1000

# Funktsiya dlya vychisleniya koda tochki
def compute_code(x, y, x_min, y_min, x_max, y_max):
    code = INSIDE
    if x < x_min:    # Sleva ot okna
        code |= LEFT
    elif x > x_max:  # Sprava ot okna
        code |= RIGHT
    if y < y_min:    # Nizhe okna
        code |= BOTTOM
    elif y > y_max:  # Vyshe okna
        code |= TOP
    return code

# Algoritm Sazerlenda-Koena dlya otsecheniya otrezkov
def cohen_sutherland_clip(x1, y1, x2, y2, x_min, y_min, x_max, y_max):
    code1 = compute_code(x1, y1, x_min, y_min, x_max, y_max)
    code2 = compute_code(x2, y2, x_min, y_min, x_max, y_max)
    accept = False

    while True:
        if code1 == 0 and code2 == 0:  # Obe tochki vnutri okna
            accept = True
            break
        elif code1 & code2 != 0:  # Obe tochki snaruji, otrezok vne okna
            break
        else:
            x, y = 0.0, 0.0
            # Vyberaem tochku, nahodyashchuyusya snaruji
            if code1 != 0:
                code_out = code1
            else:
                code_out = code2

            # Naydemy peresechenie s granitsami okna
            if code_out & TOP:  # Peresechenie s verhney granitsey
                x = x1 + (x2 - x1) * (y_max - y1) / (y2 - y1)
                y = y_max
            elif code_out & BOTTOM:  # Peresechenie s nizhney granitsey
                x = x1 + (x2 - x1) * (y_min - y1) / (y2 - y1)
                y = y_min
            elif code_out & RIGHT:  # Peresechenie s pravoy granitsey
                y = y1 + (y2 - y1) * (x_max - x1) / (x2 - x1)
                x = x_max
            elif code_out & LEFT:  # Peresechenie s levoy granitsey
                y = y1 + (y2 - y1) * (x_min - x1) / (x2 - x1)
                x = x_min

            # Zamenim tochku snaruji na tochku peresecheniya i pereschitaem kod
            if code_out == code1:
                x1, y1 = x, y
                code1 = compute_code(x1, y1, x_min, y_min, x_max, y_max)
            else:
                x2, y2 = x, y
                code2 = compute_code(x2, y2, x_min, y_min, x_max, y_max)

    if accept:
        return x1, y1, x2, y2
    else:
        return None

# Funktsiya dlya vizualizatsii otsecheniya linii
def draw_plot(lines, x_min, y_min, x_max, y_max):
    fig, ax = plt.subplots()

    # Risuyem okno otsecheniya
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min], 'k-', lw=2)

    # Risuyem otrezki do otsecheniya
    for line in lines:
        x1, y1, x2, y2 = line
        ax.plot([x1, x2], [y1, y2], 'r--', label='Do otsecheniya')

    # Otsechenie linii
    for line in lines:
        result = cohen_sutherland_clip(*line, x_min, y_min, x_max, y_max)
        if result:
            x1, y1, x2, y2 = result
            ax.plot([x1, x2], [y1, y2], 'g-', lw=2, label='Posle otsecheniya')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Otsechenie otrezkov algoritmom Sazerlenda-Koena')
    plt.grid(True)
    plt.show()

# Primer ispolzovaniya
if __name__ == "__main__":
    # Zadaem okno otsecheniya
    x_min, y_min = 10, 10
    x_max, y_max = 100, 100

    # Otrezki dlya otsecheniya
    lines = [
        (5, 5, 120, 120),
        (50, 50, 60, 70),
        (70, 80, 120, 140),
        (10, 110, 110, 10),
        (0, 50, 200, 50)
    ]

    # Vizualizatsiya
    draw_plot(lines, x_min, y_min, x_max, y_max)
```

### 5 Реализация алгоритма Цирруса-Бека

```python
import numpy as np
import matplotlib.pyplot as plt

# Funktsiya dlya vychisleniya skalyarnogo proizvedeniya dvuh vektorov
def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]

# Algoritm Cirrus-Beka dlya otsecheniya otrezkov
def cyrus_beck_clip(line_start, line_end, polygon):
    d = np.array(line_end) - np.array(line_start)  # Vektor napravleniya otrezka
    t_enter = 0  # Parametr t na vkhode
    t_exit = 1   # Parametr t na vykhode

    for i in range(len(polygon)):
        # Naidemy normal k tekushemu rebru polygon
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]
        edge = np.array(p2) - np.array(p1)
        normal = np.array([-edge[1], edge[0]])  # Perpendikulyarnyi vektor (normal)

        # Vycheslyaem vektor, vedushchiy ot starta otrezka do tochki p1
        w = np.array(line_start) - np.array(p1)

        # Vycheslyaem skalyarnye proizvedeniya
        numerator = -dot_product(w, normal)
        denominator = dot_product(d, normal)

        if denominator != 0:
            t = numerator / denominator
            if denominator > 0:  # Vkhod v polygon
                t_enter = max(t_enter, t)
            else:  # Vykhod iz polygona
                t_exit = min(t_exit, t)

            if t_enter > t_exit:
                return None  # Otrezok ne vidim

    if t_enter <= t_exit:
        # Vycheslyaem tochki peresecheniya s polygonom
        clipped_start = line_start + t_enter * d
        clipped_end = line_start + t_exit * d
        return clipped_start, clipped_end
    return None

# Funktsiya dlya vizualizatsii otsecheniya otrezka
def draw_plot(lines, polygon):
    fig, ax = plt.subplots()

    # Risuyem polygon
    polygon.append(polygon[0])  # Zamykayem polygon
    polygon = np.array(polygon)
    ax.plot(polygon[:, 0], polygon[:, 1], 'k-', lw=2)

    # Risuyem otrezki do otsecheniya
    for line in lines:
        line_start, line_end = line
        ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], 'r--', label='Do otsecheniya')

    # Otsechenie otrezkov
    for line in lines:
        result = cyrus_beck_clip(np.array(line[0]), np.array(line[1]), polygon[:-1].tolist())
        if result:
            clipped_start, clipped_end = result
            ax.plot([clipped_start[0], clipped_end[0]], [clipped_start[1], clipped_end[1]], 'g-', lw=2, label='Posle otsecheniya')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Otsechenie otrezkov algoritmom Cirrus-Beka')
    plt.grid(True)
    plt.show()

# Primer ispolzovaniya
if __name__ == "__main__":
    # Zadaem polygon (vypukly)
    polygon = [
        [10, 10],
        [100, 30],
        [90, 100],
        [30, 90]
    ]

    # Otrezki dlya otsecheniya
    lines = [
        ([0, 0], [50, 50]),
        ([20, 80], [80, 20]),
        ([60, 60], [120, 120]),
        ([0, 100], [100, 0]),
        ([70, 10], [70, 120])
    ]

    # Vizualizatsiya
    draw_plot(lines, polygon)

```

### 6 Алгоритм заполнения замкнутых областей посредством "затравки"

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def create_polygon_image(vertices, shape=(100, 100)):
    fig, ax = plt.subplots()
    fig.set_size_inches(shape[0] / fig.dpi, shape[1] / fig.dpi)
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    ax.invert_yaxis()
    ax.axis('off')

    # Рисуем многоугольник
    polygon = Polygon(vertices, closed=True, edgecolor='black', facecolor='white')
    ax.add_patch(polygon)

    # Преобразуем в массив
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8').reshape(shape[0], shape[1], 4)
    plt.close(fig)

    return image[:, :, :3].copy()

def is_background(color, threshold=68):
    # Считаем белыми пиксели с яркостью выше 68
    return np.mean(color) > threshold

def boundary_fill(image, x, y, fill_color):
    if not is_background(image[x, y]):
        return

    stack = [(x, y)]

    while stack:
        cx, cy = stack.pop()
        if is_background(image[cx, cy]):
            image[cx, cy] = fill_color

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1] and is_background(image[nx, ny]):
                    stack.append((nx, ny))

# Определяем вершины 7-угольника (хочу вот семиугольник закрасить)
vertices = [(30, 20), (70, 15), (90, 40), (80, 70), (50, 90), (20, 70), (10, 40)]
image = create_polygon_image(vertices)

fill_color = np.array([139, 0, 0], dtype=np.uint8)  # ЦВЕТ КРОВИ

# Убираем темные серые пиксели между границей и заливкой
gray_threshold = 100
image[np.all((image[:, :, 0] < gray_threshold) & 
             (image[:, :, 1] < gray_threshold) & 
             (image[:, :, 2] < gray_threshold), axis=-1)] = [255, 255, 255]

# Отображаем исходное изображение
plt.subplot(1, 2, 1)
plt.title("Исходное изображение")
plt.imshow(image)

# Применяем Boundary Fill с начальной точкой внутри многоугольника
boundary_fill(image, 50, 50, fill_color)

# Отображаем результат
plt.subplot(1, 2, 2)
plt.title("После Boundary Fill")
plt.imshow(image)
plt.show()

```

### 7 Алгоритм заполнения замкнутых областей посредством горизонтального сканирования

```python
import matplotlib.pyplot as plt
import numpy as np

def fill_polygon(vertices):
    # Sozdaniye pustogo polya
    x_min, x_max = min(vertices[:, 0]), max(vertices[:, 0])
    y_min, y_max = min(vertices[:, 1]), max(vertices[:, 1])
    
    # Spisok 4 zapolneniya
    fill_points = []
    
    # Prohod po gorizontal lines
    for y in range(int(y_min), int(y_max) + 1):
        intersections = []
        
        # Nahodim peresecheniya s ryobrami
        for i in range(len(vertices)):
            v1, v2 = vertices[i], vertices[(i + 1) % len(vertices)]
            if (v1[1] > y) != (v2[1] > y):
                x = (v2[0] - v1[0]) * (y - v1[1]) / (v2[1] - v1[1]) + v1[0]
                intersections.append(x)
        
        # Sortirue, peresecheniya
        intersections.sort()
        
        # Zapolnyaem Oblast'
        for i in range(0, len(intersections), 2):
            fill_points.append((intersections[i], y))
            fill_points.append((intersections[i + 1], y))
    
    return fill_points

# Vershini
vertices = np.array([(1, 1), (5, 0.5), (4, 4), (2, 3), (1, 4)])
fill_points = fill_polygon(vertices)

# Visualizatsia
plt.fill(vertices[:, 0], vertices[:, 1], 'lightgrey')
plt.xlim(0, 6)
plt.ylim(0, 5)
plt.show()
```

### РК1. Подзоров. Cравнение производительности алгоритма заполнения многоугольников по граничным точками и метода из библиотеки pygame.

### pygame

```python
import pygame
import sys
import time
import random

def calculate_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    area = abs(area) / 2.0
    return area

t1 = time.time()
pygame.init()

width, height = 800, 600
screen = pygame.display.set_mode((width, height))


polygon_points = [(random.randint(0, width - 30), random.randint(0, height - 30)) for _ in range(10)]
area = calculate_area(polygon_points)
for i in range(1000):
    pygame.draw.polygon(screen, (0, 0, 0), polygon_points)
    #pygame.display.flip()

pygame.quit()
t2 = time.time()
time_e = t2 - t1
print(f"Время работы: {time_e}")
print(f"Площадь многоугольника: {area}")
sys.exit()
```

### Метод граничных точек

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
from matplotlib.pyplot import imshow
from PIL import Image
import numpy as np
import time
import random

def calculateX(y, P1, P2):
    try:
        k = (P2.y - P1.y)/(P2.x-P1.x)
    except ZeroDivisionError:
        return
    else:
        return (y-P1.y)/k + P1.x


def isHorizontal(P1, P2):
    return P1.y == P2.y

def getMaxMinY(points):
    maxy = points[0].y
    miny = points[0].y
    for p in points:
        if maxy < p.y:
            maxy = p.y
        if miny > p.y:
            miny = p.y
    return maxy, miny

def draw(img, y, x1, x2):
    for x in range(round(x1), round(x2)):
        img.putpixel((x, y), (0, 0, 0))

def calculate_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i].x * points[j].y
        area -= points[j].x * points[i].y
    area = abs(area) / 2.0
    return area

def main():
    t1 = time.time()
    img = Image.new('RGB', (800, 600), 'white')
    imshow(np.asarray(img))
    points = [Point(random.randint(0, 770), random.randint(0,570)) for _ in range(10)]
    area = calculate_area(points)
    maxy, miny = getMaxMinY(points)
    for _ in range(1000):
            for y in range(miny, maxy+1):
             border = []
             for i in range(len(points)):
                P1=points[i]
                P2=points[(i+1)%(len(points))]
                if not isHorizontal(P1, P2) and y <= max(P1.y, P2.y) and y > min(P1.y, P2.y) and P1.x != P2.x:
                     x = calculateX(y, P1, P2)
                     if x is not None:
                        border.append(x)
            border.sort()
            if len(border) >= 2:
                for pi in range(0, len(border), 1):  
                     if pi + 1 < len(border):  
                         draw(img, y, border[pi], border[pi + 1])
    
   

    t2 = time.time()
    print(f'Время выполнения: {t2 - t1}')
    print(f'Площадь многоугольника: {area}')
    
main()
```

### 8 Вращение

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import imageio

# Параметры GIF
gif_filename = 'kruchu-verchu.gif'
frames = []
num_frames = 60  # Количество кадров

# Функция для поворота точки в 3D
def rotate(point, angle_x, angle_y, angle_z):
    # Углы поворота в радианах
    ax, ay, az = np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)
    
    # Матрицы вращения
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    
    # Применение вращения
    rotated_point = Rz @ Ry @ Rx @ point
    return rotated_point

# Генерация случайных точек и создание выпуклой оболочки
num_points = 50  # Количество случайных точек
points = np.random.uniform(-1, 1, (num_points, 3))

# Создание выпуклой оболочки
hull = ConvexHull(points)

# Создание фигуры
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Генерация случайных цветов для каждой грани
colors = np.random.rand(len(hull.simplices), 3)

# Создание анимации вращения
for i in range(num_frames):
    ax.clear()
    ax.set_title("Palanes")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Углы поворота
    angle_x = i * 6
    angle_y = i * 3
    angle_z = i * 2

    # Поворачиваем все точки
    rotated_points = np.array([rotate(p, angle_x, angle_y, angle_z) for p in points])

    # Рисуем грани фигуры
    for idx, simplex in enumerate(hull.simplices):
        triangle = rotated_points[simplex]
        ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                        color=colors[idx], edgecolor='k', alpha=0.8)

    # Сохранение текущего кадра
    plt.draw()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)

# Запись в GIF
imageio.mimsave(gif_filename, frames, fps=15)
print(f'GIF сохранен в файл: {gif_filename}')
```

### 9 Вращение в 240 fps

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import imageio

# Параметры GIF
gif_filename = 'kruchu v 240 fps.gif'
frames = []
num_frames = 240  # Количество кадров для 1 секунды анимации

# Функция для поворота точки в 3D
def rotate(point, angle_x, angle_y, angle_z):
    ax, ay, az = np.radians(angle_x), np.radians(angle_y), np.radians(angle_z)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx @ point

# Генерация случайных точек и создание выпуклой оболочки
num_points = 50
points = np.random.uniform(-1, 1, (num_points, 3))
hull = ConvexHull(points)

# Цвета для граней
colors = np.random.rand(len(hull.simplices), 3)

# Создание фигуры
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Генерация анимации вращения
for i in range(num_frames):
    ax.clear()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.axis('on')

    # Углы поворота
    angle_x = i * 1.5  # Скорость вращения по X
    angle_y = i * 0.75  # Скорость вращения по Y
    angle_z = i * 0.5  # Скорость вращения по Z

    # Поворачиваем точки
    rotated_points = np.array([rotate(p, angle_x, angle_y, angle_z) for p in points])

    # Отрисовка грани с черными рёбрами
    for idx, simplex in enumerate(hull.simplices):
        triangle = rotated_points[simplex]
        ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], 
                        color=colors[idx], edgecolor='k', alpha=0.8)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)

# Сохранение в GIF с 240 fps
imageio.mimsave(gif_filename, frames, fps=240)
print(f'GIF сохранен в файл: {gif_filename}')
```
