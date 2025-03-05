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
    x_values = [x0]
    y_values = [y0]
    while x0 < xn:
        k1 = f(x0, y0, expr)
        k2 = f(x0 + h, y0 + h * k1, expr)
        y0 += (h / 2) * (k1 + k2)
        x0 += h
        x_values.append(x0)
        y_values.append(y0)
    return x_values, y_values

# Método de Runge-Kutta de cuarto orden
def runge_kutta(f, x0, y0, h, xn, expr):
    x_values = [x0]
    y_values = [y0]
    while x0 < xn:
        k1 = h * f(x0, y0, expr)
        k2 = h * f(x0 + h/2, y0 + k1/2, expr)
        k3 = h * f(x0 + h/2, y0 + k2/2, expr)
        k4 = h * f(x0 + h, y0 + k3, expr)
        y0 += (k1 + 2*k2 + 2*k3 + k4) / 6
        x0 += h
        x_values.append(x0)
        y_values.append(y0)
    return x_values, y_values
# newton rambson

# Función para resolver y graficar
def resolver():
    try:
        expr = sp.sympify(equation_entry.get())
        x0 = float(x0_entry.get())
        y0 = float(y0_entry.get())
        h = float(h_entry.get())
        xn = float(xn_entry.get())
        metodo = metodo_var.get()
        
        if metodo == 'Euler':
            x_vals, y_vals = euler(f, x0, y0, h, xn, expr)
        elif metodo == 'Euler Mejorado':
            x_vals, y_vals = euler_mejorado(f, x0, y0, h, xn, expr)
        elif metodo == 'Runge-Kutta':
            x_vals, y_vals = runge_kutta(f, x0, y0, h, xn, expr)
        else:
            messagebox.showerror('Error', 'Seleccione un método')
            return
        
        # Guardar solo los primeros 10 resultados
        guardar_resultados(metodo, x_vals[:10], y_vals[:10], x0, y0, h, xn, expr)
        
        # Mostrar los primeros 10 resultados en un cuadro de diálogo
        mostrar_resultados(x_vals[:10], y_vals[:10])

        plt.plot(x_vals, y_vals, marker='o', label=metodo)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Método ' + metodo)
        plt.legend()
        plt.grid()
        plt.show()
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

# Función para reiniciar los campos de entrada
def reiniciar():
    equation_entry.delete(0, tk.END)
    x0_entry.delete(0, tk.END)
    y0_entry.delete(0, tk.END)
    h_entry.delete(0, tk.END)
    xn_entry.delete(0, tk.END)
    metodo_var.set('')
    # para que el grafico se reinicie
    plt.close()


# Configuración de la ventana
root = tk.Tk()
root.title('Solución de EDOs')

metodo_var = tk.StringVar()

# Entradas con ejemplos al lado
tk.Label(root, text='Ecuación diferencial dy/dx=', font=("Arial", 20)).grid(row=0, column=0, sticky='e')
equation_entry = tk.Entry(root, width=30, font=("Arial", 18))
equation_entry.grid(row=0, column=1, sticky='w')
tk.Label(root, text='Ej: x+y', font=("Arial", 18)).grid(row=0, column=2, sticky='w')

tk.Label(root, text='x0:', font=("Arial", 20)).grid(row=1, column=0, sticky='e')
x0_entry = tk.Entry(root, font=("Arial", 18))
x0_entry.grid(row=1, column=1, sticky='w')
tk.Label(root, text='Ej: 0', font=("Arial", 18)).grid(row=1, column=2, sticky='w')

tk.Label(root, text='y0:', font=("Arial", 20)).grid(row=2, column=0, sticky='e')
y0_entry = tk.Entry(root, font=("Arial", 18))
y0_entry.grid(row=2, column=1, sticky='w')
tk.Label(root, text='Ej: 1', font=("Arial", 18)).grid(row=2, column=2, sticky='w')

tk.Label(root, text='h:', font=("Arial", 20)).grid(row=3, column=0, sticky='e')
h_entry = tk.Entry(root, font=("Arial", 18))
h_entry.grid(row=3, column=1, sticky='w')
tk.Label(root, text='Ej: 0.1', font=("Arial", 18)).grid(row=3, column=2, sticky='w')

tk.Label(root, text='xn:', font=("Arial", 20)).grid(row=4, column=0, s