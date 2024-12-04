"""Esperimenti per GUI"""

import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def create_matplotlib_plot() -> (np.ndarray, int, int):
    # genera una Figure con un plot a caso
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y, label="sin(x)")
    ax.set_title("Example Plot")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.legend()

    # dumpa la figure nella canvas
    canvas = FigureCanvas(fig)
    canvas.draw()

    # estrai il color buffer dalla canvas
    buf = canvas.buffer_rgba() # ritorna memoryview
    image = np.asarray(buf) # converti in readonly np.ndarray
    image = image.astype(np.float32) / 255 # crea una copia da int[0, 255] a np.float32 [0, 1]
    width, height = canvas.get_width_height()

    return image, width, height

def save_callback():
    print("Save Clicked")

def main() -> None:
    dpg.create_context()
    dpg.create_viewport()
    dpg.setup_dearpygui()

    # Create the matplotlib plot as a texture
    plot_image, plot_width, plot_height = create_matplotlib_plot()

    # Register the image as a texture
    with dpg.texture_registry():
        dpg.add_raw_texture(plot_width, plot_height, plot_image, format=dpg.mvFormat_Float_rgba, tag="plot_id")

    with dpg.window(label="Example Window"):
        dpg.add_text("Hello world")
        dpg.add_button(label="Save", callback=save_callback)
        dpg.add_input_text(label="string")
        dpg.add_slider_float(label="float")

        # Draw the image (matplotlib plot) in the window
        dpg.add_image("plot_id") # fa schifissimo, forse meglio fare i plot diretto con dearpygui

    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == "__main__":
    main()