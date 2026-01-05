import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import math


class SpatialPoint:
    def __init__(self, x1, x2):
        self.x1 = x1 #добавляем координаты
        self.x2 = x2

    def __repr__(self):
        return f"SpatialPoint({self.x1}, {self.x2})"

    def to_tuple(self):
        return (self.x1, self.x2)


class MaterialPoint(SpatialPoint):  # Наследование от SpatialPoint
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
        self.points = []  # Список SpatialPoint over time

    def add_point(self, point):
        self.points.append(point)

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


class CircleBody(Body):  # Конкретная реализация для окружности
    def __init__(self, center_x1, center_x2, radius, num_points=100):
        self.center = SpatialPoint(center_x1, center_x2)
        self.radius = radius
        self.points = []
        # Генерируем точки на окружности (полная, но в первой четверти по положению)
        angles = np.linspace(0, 2 * np.pi, num_points)
        for theta in angles:
            x1 = center_x1 + radius * np.cos(theta)
            x2 = center_x2 + radius * np.sin(theta)
            if x1 > 0 and x2 > 0:  # Только первая четверть, но поскольку центр в (5,5), все точки в первой
                self.points.append(MaterialPoint(x1, x2))

    def get_points(self):
        return self.points

    def get_initial_shape(self):
        return [p.to_tuple() for p in self.points]

    def get_deformed_shape(self):
        return [(p.current_x1, p.current_x2) for p in self.points]

    def update_points(self, integrator, t_start, t_end, h):
        for point in self.points:
            # Полиморфизм: integrator.integrate работает с MaterialPoint
            integrator.integrate(point, t_start, t_end, h)


class Integrator(ABC):
    @abstractmethod
    def integrate(self, point, t_start, t_end, h):
        pass


class RungeKutta33(Integrator):
    def __init__(self, A_func, B_func):
        self.A = A_func  # lambda t: math.log(t)
        self.B = B_func  # lambda t: math.exp(t)

    def integrate(self, point, t_start, t_end, h):
        # Интегрируем от t_start до t_end, обновляем позицию point
        t = t_start
        while t < t_end:
            # Для x1
            k11 = -self.A(t) * point.current_x1
            k21 = -self.A(t + h / 2) * (point.current_x1 + h * k11 / 2)
            k31 = -self.A(t + h) * (point.current_x1 + h * (-k11 + 2 * k21))
            dx1 = h * (k11 / 6 + 2 * k21 / 3 + k31 / 6)

            # Для x2
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

    def plot_trajectories(self, trajectories, filename):
        plt.figure()
        for traj in trajectories:
            plt.plot(traj.get_x1_list(), traj.get_x2_list())
        plt.title("Графики траекторий")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    def plot_shapes(self, initial, deformed, filename):
        plt.figure(figsize=(10, 8))

        init_x1, init_x2 = zip(*initial)
        def_x1, def_x2 = zip(*deformed)

        plt.plot(init_x1, init_x2, label="Начальная форма", linewidth=2.5, color='blue')
        plt.plot(def_x1, def_x2, label="Деформированная форма (t=1.5)", linewidth=2.5, color='red')

        plt.title("Начальная и деформированная форма тела")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.7)

        # Хороший масштаб для деформации при t=1.5
        plt.gca().set_aspect(1 / 1, adjustable='box')  # y в 5 раз "растянутее" визуально

        # Пределы осей — всё точно влезет
        plt.xlim(0, 10)
        plt.ylim(0, 55)  # При t=1.5 максимум y ≈ 5 + 3 * exp(exp(1.5)-exp(1)) ≈ 40

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()

    def plot_velocity_and_streamlines(self, t, area_min=1, area_max=10, grid_size=20, filename_vel="vel.png",
                                      filename_stream="stream.png"):
        x = np.linspace(area_min, area_max, grid_size)
        y = np.linspace(area_min, area_max, grid_size)
        X, Y = np.meshgrid(x, y)
        V1 = -self.A(t) * X
        V2 = self.B(t) * Y

        # Velocity field
        plt.figure()
        plt.quiver(X, Y, V1, V2)
        plt.title(f"Поле скоростей в t={t}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.savefig(filename_vel)
        plt.close()

        # Streamlines
        plt.figure()
        plt.streamplot(X, Y, V1, V2)
        plt.title(f"Линии тока в t={t}")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.savefig(filename_stream)
        plt.close()


# --- main.py ---
if __name__ == "__main__":
    # Функции A и B как лямбда
    A = lambda t: math.log(t)
    B = lambda t: math.exp(t)

    # Параметры
    t_start = 1.0
    t_end = 1.5  # Уменьшаем время — деформация всё равно сильная, но стабильная
    h = 0.005  # Меньший шаг для большей точности
    times = [1.0, 1.25, 1.5]  # Моменты времени соответственно

    # Создаем тело
    body = CircleBody(center_x1=5, center_x2=5, radius=3, num_points=100)

    # Интегратор
    integrator = RungeKutta33(A, B)

    # Траектории для нескольких точек (выберем 5)
    representative_points = body.get_points()[::20]  # Каждые 20-е
    trajectories = []
    for point in representative_points:
        traj = Trajectory()
        traj.add_point(SpatialPoint(point.current_x1, point.current_x2))  # Начальная
        t = t_start
        while t < t_end:
            # Интегрируем на один шаг, но добавляем каждый 10-й для простоты
            integrator.integrate(point, t, t + h, h)
            if int((t - t_start) / h) % 10 == 0:
                traj.add_point(SpatialPoint(point.current_x1, point.current_x2))
            t += h
        trajectories.append(traj)
        # Сброс позиции точки для deformed (но для траекторий мы интегрировали копию? Нет, point updated, но для deformed мы используем после
        # Подождите, для траекторий я updated point, но для тела нужно отдельно.
        # Ошибка: лучше клонировать points для траекторий.

    # Пересоздаем тело для чистоты
    body = CircleBody(center_x1=5, center_x2=5, radius=3, num_points=100)
    initial_shape = body.get_initial_shape()

    # Обновляем тело до t_end
    body.update_points(integrator, t_start, t_end, h)
    deformed_shape = body.get_deformed_shape()

    # Плоттер
    plotter = Plotter(A, B)
    plotter.plot_trajectories(trajectories, "trajectories.png")
    plotter.plot_shapes(initial_shape, deformed_shape, "shapes.png")

    # Поля в разные времена
    for idx, t in enumerate(times):
        plotter.plot_velocity_and_streamlines(t, filename_vel=f"velocity_t{idx}.png",
                                              filename_stream=f"streamlines_t{idx}.png")

    print("Графики траекторий: trajectories.png")
    print("Начальная и деформированная форма: shapes.png")
    print("Поля скоростей и линии тока: velocity_t*.png и streamlines_t*.png для t=1.0,1.5,2.0")
