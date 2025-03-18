import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import json
import os

# Función para evaluar la ecuación diferencial
def f(x, y, expr):
    return expr.subs({'x': x, 'y': y})

# Método de Euler
def euler(f, x0, y0, h, xn, expr):
    x_values = [x0]
    y_values = [y0]
    while x0 < xn:
        y0 += h * f(x0, y0, expr) #explicame que hace aqui 
        x0 += h
        x_values.append(x0)
        y_values.append(y0)
    return x_values, y_values

# Método de Euler Mejorado
def euler_mejorado(f, x0, y0, h, xn, expr):
    if h <= 0 or x0 >= xn:
        messagebox.showerror('Error', 'El valor de h debe ser positivo y x0 debe ser menor que xn')
        return [], []

    # Inicialización de listas para almacenar resultados
    x_values = [x0]
    y_values = [y0]
    y_pred_values = []  # (yn+1)*
    x_next_values = []  # xn+1
    y_next_values = []  # yn+1
    k1_values = []
    k2_values = []

    while x0 < xn:
        k1 = f(x0, y0, expr)  # k1 = f(xn, yn)
        y_pred = y0 + h * k1  # (yn+1)* = yn + h * k1
        k2 = f(x0 + h, y_pred, expr)  # k2 = f(xn+1, (yn+1)*)

        # Cálculo del nuevo valor de y usando Euler Mejorado
        y_siguiente = y0 + (h / 2) * (k1 + k2)  # yn+1

        # Guardar valores en listas
        k1_values.append(k1)
        y_pred_values.append(y_pred)
        k2_values.append(k2)
        x_next = x0 + h  # xn+1
        x_next_values.append(x_next)
        y_next_values.append(y_siguiente)

        # Actualizar x0 e y0 para la siguiente iteración
        x0 = x_next
        y0 = y_siguiente

        x_values.append(x0)
        y_values.append(y0)

    # Crear ventana de resultados
    resultados_window = tk.Toplevel()
    resultados_window.title("Resultados Euler Mejorado")

    # Configurar la tabla
    tree = ttk.Treeview(resultados_window, columns=("Paso", "xn", "yn", "(yn+1)*", "xn+1", "yn+1", "k1", "k2"), show='headings')
    tree.heading("Paso", text="Paso")
    tree.heading("xn", text="xn")
    tree.heading("yn", text="yn")
    tree.heading("(yn+1)*", text="(yn+1)*")
    tree.heading("xn+1", text="xn+1")
    tree.heading("yn+1", text="yn+1")
    tree.heading("k1", text="k1")
    tree.heading("k2", text="k2")
    tree.pack(expand=True, fill='both')

    # Insertar datos en la tabla
    for paso, (xn, yn, y_pred, xn1, yn1, k1, k2) in enumerate(zip(x_values[:-1], y_values[:-1], y_pred_values, x_next_values, y_next_values, k1_values, k2_values)):
        tree.insert("", "end", values=(paso, round(xn, 4), round(yn, 4), round(y_pred, 4), round(xn1, 4), round(yn1, 4), round(k1, 4), round(k2, 4)))

    # Graficar los resultados
    plt.plot(x_values, y_values, marker='o', label='Euler Mejorado')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Método de Euler Mejorado')
    plt.legend()
    plt.grid()
    plt.show()

    return x_values, y_values


# Método de Runge-Kutta de cuarto orden
def runge_kutta(f, x0, y0, h, xn, expr):
    x_values = [round(x0, 1)]
    y_values = [y0]
    resultados = []  # Guardar valores para la tabla
    paso = 0

    while x0 < xn:
        k1 = h * f(x0, y0, expr)
        k2 = h * f(x0 + h/2, y0 + k1/2, expr)
        k3 = h * f(x0 + h/2, y0 + k2/2, expr)
        k4 = h * f(x0 + h, y0 + k3, expr)

        y_siguiente = y0 + (k1 + 2*k2 + 2*k3 + k4) * h / 6
        resultados.append((paso, round(x0, 1), y0, k1, k2, k3, k4, y_siguiente))

        # Actualizar valores
        y0 = y_siguiente
        x0 += h
        x_values.append(round(x0, 1))
        y_values.append(y0)
        paso += 1

    # Crear ventana de resultados
    resultados_window = tk.Toplevel()
    resultados_window.title("Resultados Runge-Kutta")

    # Configurar la tabla
    tree = ttk.Treeview(resultados_window, columns=("Paso", "x", "y", "k1", "k2", "k3", "k4", "y_siguiente"), show='headings')
    tree.heading("Paso", text="Paso")
    tree.heading("x", text="x")
    tree.heading("y", text="y")
    tree.heading("k1", text="k1")
    tree.heading("k2", text="k2")
    tree.heading("k3", text="k3")
    tree.heading("k4", text="k4")
    tree.heading("y_siguiente", text="y_siguiente")
    tree.pack(expand=True, fill='both')

    for resultado in resultados:
        tree.insert("", "end", values=resultado)

    # Graficar los resultados
    plt.plot(x_values, y_values, marker='o', label='Runge-Kutta')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Método de Runge-Kutta')
    plt.legend()
    plt.grid()
    plt.show() 

# newton rambson
def newton_raphson(f, x0, tol, max_iter=100):
    x = sp.symbols('x')
    f_prime = sp.diff(f, x)  # Derivada de f
    f_lambdified = sp.lambdify(x, f, "numpy")  # Convertir f a función evaluable
    f_prime_lambdified = sp.lambdify(x, f_prime, "numpy")  # Convertir f' a función evaluable
    
    x_n = x0  # Valor inicial
    
    # Crear ventana de resultados
    resultados_window = tk.Toplevel()
    resultados_window.title("Resultados Newton-Raphson")

    # Configurar la tabla
    tree = ttk.Treeview(resultados_window, columns=("Iteración", "x_n", "f(x_n)", "f'(x_n)", "x_nuevo", "Error"), show='headings')
    tree.heading("Iteración", text="Iteración")
    tree.heading("x_n", text="x_n")
    tree.heading("f(x_n)", text="f(x_n)")
    tree.heading("f'(x_n)", text="f'(x_n)")
    tree.heading("x_nuevo", text="x_nuevo")
    tree.heading("Error", text="Error")
    tree.pack(expand=True, fill='both')

    x_vals = []
    y_vals = []


    for i in range(max_iter):
        f_val = f_lambdified(x_n)
        f_prime_val = f_prime_lambdified(x_n)

        if abs(f_prime_val) < 1e-12:  # Evitar división por cero
            messagebox.showerror("Error", "Derivada cercana a cero, posible punto crítico o falta de convergencia.")
            return

        x_n1 = x_n - f_val / f_prime_val  # Fórmula de Newton-Raphson
        error = abs(x_n1 - x_n)  # Calcular el error

        # Insertar valores en la tabla
        tree.insert("", "end", values=(i, round(x_n, 6), round(f_val, 6), round(f_prime_val, 6), round(x_n1, 6), round(error, 6)))

        x_vals.append(x_n)
        y_vals.append(f_val)

        if abs(x_n1 - x_n) < tol:  # Criterio de convergencia
            break  # Salir del bucle si la tolerancia se cumple

        x_n = x_n1  # Actualizar x_n
    tree.insert("", "end", values=(round(x_n1, 6),0,0,0,0))

    # Graficar los resultados
    plt.plot(x_vals, y_vals, marker='o', label='Newton-Raphson')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Método de Newton-Raphson')
    plt.legend()
    plt.grid()
    plt.show()

# Función para resolver y graficar
def resolver():
    try:
        expr = sp.sympify(equation_entry.get())
        x0 = float(x0_entry.get())
        metodo = metodo_var.get()

        if metodo != 'Newton-Raphson':
            y0 = float(y0_entry.get())
            h = float(h_entry.get())
            xn = float(xn_entry.get())
        else:
            tol = float(tol_entry.get()) 
        
        if metodo == 'Euler':
            euler(f, x0, y0, h, xn, expr)
        elif metodo == 'Euler Mejorado':
            euler_mejorado(f, x0, y0, h, xn, expr)
        elif metodo == 'Runge-Kutta':
            runge_kutta(f, x0, y0, h, xn, expr)
        elif metodo == 'Newton-Raphson':
            newton_raphson(expr, x0, tol)
        else:
            messagebox.showerror('Error', 'Seleccione un método')
            return
        


    except Exception as e:
        messagebox.showerror('Error', str(e))

# Función para guardar los resultados
def guardar_resultados(metodo, x_vals, y_vals, x0, y0, h, xn, expr):
    resultados = {
        "metodo": metodo,
        "ecuacion": str(expr),
        "x0": float(x0),
        "y0": float(y0),
        "h": float(h),
        "xn": float(xn),
        "resultados": [{"x": float(x_val), "y": float(y_val)} for x_val, y_val in zip(x_vals, y_vals)]
    }
    
    # Leer el contenido actual del archivo JSON
    if os.path.exists('resultados.json'):
        with open('resultados.json', 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    # Agregar el nuevo resultado al array
    data.append(resultados)
    
    # Escribir el array completo de vuelta al archivo
    with open('resultados.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    resultados = {
        "metodo": metodo,
        "ecuacion": str(expr),
        "x0": float(x0),
        "y0": float(y0),
        "h": float(h),
        "xn": float(xn),
        "resultados": [{"x": float(x_val), "y": float(y_val)} for x_val, y_val in zip(x_vals, y_vals)]
    }
    
    # Leer el contenido actual del archivo JSON
    if os.path.exists('resultados.json'):
        with open('resultados.json', 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    # Agregar el nuevo resultado al array
    data.append(resultados)
    
    # Escribir el array completo de vuelta al archivo
    with open('resultados.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    resultados = {
        "metodo": metodo,
        "ecuacion": str(expr),
        "x0": float(x0),
        "y0": float(y0),
        "h": float(h),
        "xn": float(xn),
        "resultados": [{"x": float(x_val), "y": float(y_val)} for x_val, y_val in zip(x_vals, y_vals)]
    }
    
    with open('resultados.json', 'a', encoding='utf-8') as file:
        json.dump(resultados, file, ensure_ascii=False, indent=4)
        file.write('\n')
def mostrar_resultados(x_vals, y_vals):
    resultados_window = tk.Toplevel(root)
    resultados_window.title("Resultados")
    
    tree = ttk.Treeview(resultados_window, columns=("x", "y"), show='headings')
    tree.heading("x", text="x")
    tree.heading("y", text="y")
    
    for x_val, y_val in zip(x_vals, y_vals):
        tree.insert("", "end", values=(x_val, y_val))
    
    tree.pack(expand=True, fill='both')

# Función de ejemplo
def ejemplo():
    # Definir los parámetros
    expr = sp.sympify('-2*y')
    x0 = 0
    y0 = 1
    h = 0.1
    xn = 5
    
    # Resolver usando cada método
    x_vals_euler, y_vals_euler = euler(f, x0, y0, h, xn, expr)
    x_vals_euler_mejorado, y_vals_euler_mejorado = euler_mejorado(f, x0, y0, h, xn, expr)
    x_vals_runge_kutta, y_vals_runge_kutta = runge_kutta(f, x0, y0, h, xn, expr)
    
    # Graficar los resultados
    plt.plot(x_vals_euler, y_vals_euler, label='Euler', marker='o')
    plt.plot(x_vals_euler_mejorado, y_vals_euler_mejorado, label='Euler Mejorado', marker='x')
    plt.plot(x_vals_runge_kutta, y_vals_runge_kutta, label='Runge-Kutta', marker='s')
    
    # Solución exacta
    x_exact = np.linspace(x0, xn, 100)
    y_exact = np.exp(-2 * x_exact)

    guardar_resultados('Solución Exacta', x_exact, y_exact, x0, y0, h, xn, expr)
    
    plt.plot(x_exact, y_exact, label='Solución Exacta', color='black', linestyle='--')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparación de Métodos para EDO: dy/dx = -2y')
    plt.legend()
    plt.grid()
    plt.show()

# Función para manejar el evento de cierre de la ventana
def on_closing():
    if messagebox.askokcancel("Salir", "¿Quieres salir del programa?"):
        root.destroy()

def actualizar_campos(*args):
    metodo = metodo_var.get()
    if metodo == 'Newton-Raphson':
        y0_label.grid_remove()
        y0_entry.grid_remove()
        h_label.grid_remove()
        h_entry.grid_remove()
        xn_label.grid_remove()
        xn_entry.grid_remove()
        tol_label.grid()
        tol_entry.grid()
    else:
        y0_label.grid()
        y0_entry.grid()
        h_label.grid()
        h_entry.grid()
        xn_label.grid()
        xn_entry.grid()
        tol_label.grid_remove()
        tol_entry.grid_remove()

# Función para reiniciar los campos de entrada
def reiniciar():
    equation_entry.delete(0, tk.END)
    x0_entry.delete(0, tk.END)
    y0_entry.delete(0, tk.END)
    h_entry.delete(0, tk.END)
    xn_entry.delete(0, tk.END)
    tol_entry.delete(0, tk.END)
    metodo_var.set('')
    # para que el grafico se reinicie
    plt.close()

# Interfaz gráfica con Tkinter
root = tk.Tk()
root.title('Solución de EDOs')

# Entradas
metodo_var = tk.StringVar()
metodo_var.trace('w', actualizar_campos)

tk.Label(root, text='Ecuación diferencial dy/dx=', font=("Arial", 20)).grid(row=0, column=0, sticky='e')
equation_entry = tk.Entry(root, width=30, font=("Arial", 18))
equation_entry.grid(row=0, column=1, sticky='w')
tk.Label(root, text='Ej: x+y', font=("Arial", 18)).grid(row=0, column=2, sticky='w')

tk.Label(root, text='x0:', font=("Arial", 20)).grid(row=1, column=0, sticky='e')
x0_entry = tk.Entry(root, font=("Arial", 18))
x0_entry.grid(row=1, column=1, sticky='w')
tk.Label(root, text='Ej: 0', font=("Arial", 18)).grid(row=1, column=2, sticky='w')

y0_label = tk.Label(root, text='y0:', font=("Arial", 20))
y0_label.grid(row=2, column=0, sticky='e')
y0_entry = tk.Entry(root, font=("Arial", 18))
y0_entry.grid(row=2, column=1, sticky='w')
tk.Label(root, text='Ej: 1', font=("Arial", 18)).grid(row=2, column=2, sticky='w')

h_label = tk.Label(root, text='h:', font=("Arial", 20))
h_label.grid(row=3, column=0, sticky='e')
h_entry = tk.Entry(root, font=("Arial", 18))
h_entry.grid(row=3, column=1, sticky='w')
tk.Label(root, text='Ej: 0.1', font=("Arial", 18)).grid(row=3, column=2, sticky='w')

xn_label = tk.Label(root, text='xn:', font=("Arial", 20))
xn_label.grid(row=4, column=0, sticky='e')
xn_entry = tk.Entry(root, font=("Arial", 18))
xn_entry.grid(row=4, column=1, sticky='w')
tk.Label(root, text='Ej: 2', font=("Arial", 18)).grid(row=4, column=2, sticky='w')

tol_label = tk.Label(root, text='tolerancia:', font=("Arial", 20))
tol_label.grid(row=5, column=0, sticky='e')
tol_entry = tk.Entry(root, font=("Arial", 18))
tol_entry.grid(row=5, column=1, sticky='w')
tk.Label(root, text='Ej: 1e-7', font=("Arial", 18)).grid(row=5, column=2, sticky='w')

# Métodos
tk.Label(root, text='Método:', font=("Arial", 20)).grid(row=6, column=0, sticky='e')
tk.Radiobutton(root, text='Euler Mejorado', variable=metodo_var, value='Euler Mejorado', font=("Arial", 18)).grid(row=7, column=1, sticky='w')
tk.Radiobutton(root, text='Runge-Kutta', variable=metodo_var, value='Runge-Kutta', font=("Arial", 18)).grid(row=8, column=1, sticky='w')
tk.Radiobutton(root, text='Newton-Raphson', variable=metodo_var, value='Newton-Raphson', font=("Arial", 18)).grid(row=9, column=1, sticky='w')

# Botones
tk.Button(root, text='Resolver', font=("Arial", 18), command=resolver).grid(row=10, column=1, sticky='w')
tk.Button(root, text='Reiniciar', font=("Arial", 18), command=reiniciar).grid(row=10, column=2, sticky='w')

# Descripción de uso
descripcion = "Ingrese la ecuación en términos de x e y. Establezca los valores iniciales (x0, y0), el paso (h) y el valor final (xn). Seleccione un método y presione 'Resolver'."
tk.Label(root, text=descripcion, font=("Arial", 16), wraplength=580, justify='left').grid(row=11, column=0, columnspan=3, pady=10)

root.mainloop()
