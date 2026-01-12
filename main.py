import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import math


# В данном коде есть логическое разделение: сущности (Point, Trajectory, Body) и действия (Integrator, Plotter).

class SpatialPoint:
    def __init__(self, x1, x2):
        self.x1 = x1  # добавляем координаты
        self.x2 = x2

    def __repr__(self):
        return f"SpatialPoint({self.x1:.3f}, {self.x2:.3f})"

    def to_tuple(self):
        return (self.x1, self.x2)


class MaterialPoint(SpatialPoint):
    # Наследование(одно из требования по курсовой, часть ООП)— один класс (дочерний) может "унаследовать" все атрибуты и методы от другого класса (родительского). Дочерний класс получает всё, что есть у родителя, и может добавить своё или переопределить. В нашем случае: родительский класс-SpatialPoint - это базовая точка в пространстве. У неё есть свои атрибуты(self.x1 и self.x2) и методы(__repr__() и to_tuple()). Дочерний класс - MaterialPoint, это уже материальная точка, которая движется во времени. Она наследует всё от SpatialPoint и добавляет своё.
    def __init__(self, initial_x1, initial_x2):
        super().__init__(initial_x1, initial_x2)
        self.current_x1 = initial_x1
        self.current_x2 = initial_x2

    def update_position(self, new_x1, new_x2):
        self.current_x1 = new_x1
        self.current_x2 = new_x2
        self.x1 = new_x1  # Обновляем базовые атрибуты для полиморфизма
        self.x2 = new_x2


class Trajectory:
    def __init__(self):
        self.points = []  # Список точек во времени

    def add_point(self, point): # Синоним для совместимости
        self.points.append(point) #Возвращает списки координат для построения графика

    def get_x1_list(self):
        return [p.x1 for p in self.points]

    def get_x2_list(self):
        return [p.x2 for p in self.points]


class Body(ABC):
    @abstractmethod
    def get_points(self):
        pass

    @abstractmethod
    def update_points(self, integrator, t_start, t_end, h):
        pass


class CircleBody(Body):
    def __init__(self, center_x=5.0, center_y=5.0, radius=3.0, num_points=100): # параметры как аргументы
        if center_x - radius <= 0 or center_y - radius <= 0:
            raise ValueError("Окружность должна быть в первой четверти!")

        angles = np.linspace(0, 2 * np.pi, num_points)
        self.points = []
        for angle in angles:
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            self.points.append(MaterialPoint(x, y))

        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius

    def get_points(self):
        return self.points

    def update_points(self, integrator, t_start, t_end, h):
        for point in self.points:
            integrator.integrate(point, t_start, t_end, h)

    def get_initial_shape(self): #Возвращает начальные координаты окружности
        return [(p.x1, p.x2) for p in self.points]

    def get_deformed_shape(self): #Возвращает текущие координаты окружности после деформации
        return [(p.current_x1, p.current_x2) for p in self.points]


class Integrator(ABC):
    @abstractmethod
    def integrate(self, point, t_start, t_end, h):
        pass


class RungeKutta33(Integrator):
    def __init__(self, A_func, B_func):
        self.A = A_func
        self.B = B_func

    def integrate(self, point, t_start, t_end, h):
        # здесь есть Полиморфизм - один и тот же код работает с объектами разных классов, если они поддерживают нужный интерфейс. Здесь: метод integrate работает с любой точкой, у которой есть нужные атрибуты (current_x1 и current_x2), независимо от того, какого она конкретно типа.
        # Интегрируем от t_start до t_end, обновляем позицию point
        t = t_start
        while t < t_end - 1e-10: # учет погрешности округления
            k11 = -self.A(t) * point.current_x1 # внутри класса RungeKutta33 лямбда-функции сохраняются как self.A и self.B (см. их ввод стр.226)
            k21 = -self.A(t + h / 2) * (point.current_x1 + h * k11 / 2)
            k31 = -self.A(t + h) * (point.current_x1 + h * (-k11 + 2 * k21))
            dx1 = h * (k11 / 6 + 2 * k21 / 3 + k31 / 6)

            k12 = self.B(t) * point.current_x2
            k22 = self.B(t + h / 2) * (point.current_x2 + h * k12 / 2)
            k32 = self.B(t + h) * (point.current_x2 + h * (-k12 + 2 * k22))
            dx2 = h * (k12 / 6 + 2 * k22 / 3 + k32 / 6)

            point.update_position(point.current_x1 + dx1, point.current_x2 + dx2)
            t += h


class Plotter:
    def __init__(self, A_func, B_func):
        self.A = A_func
        self.B = B_func
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False

    def plot_trajectories(self, trajectories):
        plt.figure(figsize=(10, 8))
        for traj in trajectories:
            plt.plot(traj.get_x1_list(), traj.get_x2_list(), linewidth=2.0)
        plt.title("Графики траекторий движения материальных точек", fontsize=14)
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True, alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_shapes(self, initial, deformed, time):
        plt.figure(figsize=(10, 8))
        init_x1, init_x2 = zip(*initial)
        def_x1, def_x2 = zip(*deformed)
        plt.plot(init_x1, init_x2, label="Начальная форма", linewidth=2.5, color='blue')
        plt.plot(def_x1, def_x2, label=f"Деформированная форма (t={time})", linewidth=2.5, color='red')
        plt.title("Начальная и деформированная форма тела", fontsize=14)
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.7)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xlim(0, 10)
        plt.ylim(0, 55)
        plt.tight_layout()
        plt.show()

    def plot_velocity_and_streamlines(self, t):
        x = np.linspace(1, 9, 20)
        y = np.linspace(1, 40, 20)
        X, Y = np.meshgrid(x, y)
        V1 = -math.log(t) * X
        V2 = math.exp(t) * Y
        magnitude = np.sqrt(V1 ** 2 + V2 ** 2)

        # Поле скоростей
        plt.figure(figsize=(10, 8))
        V1_norm = V1 / (magnitude + 1e-8)
        V2_norm = V2 / (magnitude + 1e-8)
        plt.quiver(X, Y, V1_norm, V2_norm, magnitude, cmap='plasma', scale=25, width=0.004)
        plt.colorbar(label='Величина скорости')
        plt.title(f"Поле скоростей при t = {t:.2f}", fontsize=14)
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.show()

        # Линии тока
        x_fine = np.linspace(1, 9, 40)
        y_fine = np.linspace(1, 40, 40)
        X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
        V1_fine = -math.log(t) * X_fine
        V2_fine = math.exp(t) * Y_fine
        magnitude_fine = np.sqrt(V1_fine ** 2 + V2_fine ** 2)

        plt.figure(figsize=(10, 8))
        plt.streamplot(X_fine, Y_fine, V1_fine, V2_fine, color=magnitude_fine, cmap='plasma', linewidth=1.8, density=2,
                       arrowstyle='->')
        plt.colorbar(label='Величина скорости')
        plt.title(f"Линии тока при t = {t:.2f}", fontsize=14)
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.show()


def compute_trajectories(body, integrator, t_start, t_end, dt, step=20, save_every=5):
    """Вычисление траекторий для подмножества точек"""
    trajectories = []
    for i in range(0, len(body.points), step):
        original_point = body.points[i]
        temp_point = MaterialPoint(original_point.x1, original_point.x2) # Исправлено: MaterialPoint вместо Point

        traj = Trajectory()
        traj.add_point(SpatialPoint(temp_point.current_x1, temp_point.current_x2)) # Исправлено: add_point

        t = t_start
        step_count = 0
        while t < t_end - 1e-10:
            integrator.integrate(temp_point, t, t + dt, dt)
            t += dt
            step_count += 1
            if step_count % save_every == 0:
                traj.add_point(SpatialPoint(temp_point.current_x1, temp_point.current_x2))

        traj.add_point(SpatialPoint(temp_point.current_x1, temp_point.current_x2))
        trajectories.append(traj)

    return trajectories


def main():
    t_start = 1.0
    t_end = 1.5
    dt = 0.005
    times = [1.0, 1.25, 1.5]

    A = lambda t: math.log(t) #Здесь мы реализуем наши функции через лямбда-функции. Это по факту для упрощения работы. Это то же самое, что если бы мы писали полноценную функцию:Pythondef A(t):; if t > 0:; return math.log(t); else:; return 0. Но вместо 5 строк — всего одна. Добавлена защита if t > 0 else 0, чтобы не было ошибки логарифма от нуля или отрицательного числа
    B = lambda t: math.exp(t)

    body = CircleBody(center_x=5.0, center_y=5.0, radius=3.0, num_points=100)
    integrator = RungeKutta33(A, B) #Эти две лямбда-функции передаются в интегратор и реализуются в методе рунге-кутты (стр.105)
    plotter = Plotter(A, B)

    print("Запуск расчёта... Окна с графиками будут появляться по очереди.")
    print("Закрывайте каждое окно, чтобы увидеть следующее.\n")

    # Траектории
    trajectories = compute_trajectories(body, integrator, t_start, t_end, dt, step=20, save_every=5)
    plotter.plot_trajectories(trajectories)

    # Деформация тела
    initial_shape = body.get_initial_shape()
    deformed_body = CircleBody(center_x=5.0, center_y=5.0, radius=3.0, num_points=100) # центр (5,5) выбрали сами, просто чтобы материально тело не пересекалось с осями и находилось на некотором расстоянии
    deformed_body.update_points(integrator, t_start, t_end, dt)
    deformed_shape = deformed_body.get_deformed_shape()
    plotter.plot_shapes(initial_shape, deformed_shape, t_end)

    # Поля и линии тока
    for t in times:
        plotter.plot_velocity_and_streamlines(t)

    print("\nВсе графики показаны! Программа завершена.")


if __name__ == "__main__":
    main()