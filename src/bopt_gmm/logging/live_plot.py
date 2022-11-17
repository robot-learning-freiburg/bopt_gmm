import dearpygui.dearpygui as dpg

HAS_DPG_CONTEXT = False

def create_dpg_context(width=900, height=450):
    global HAS_DPG_CONTEXT
    if not HAS_DPG_CONTEXT:
        dpg.create_context()
        dpg.create_viewport(title='Live Vis', width=900, height=450)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        HAS_DPG_CONTEXT = True

is_dpg_running   = dpg.is_dearpygui_running
render_dpg_frame = dpg.render_dearpygui_frame

class LivePlot():
    def __init__(self, plot_name, values, n_points=1000):
        self._data = {v: [] for v in values.keys()}
        self._sorted_items = sorted(values.items(), key=lambda t: t[1])
        self._x_coords = list(range(-n_points, 0))

        with dpg.window(label=plot_name):
            # create plot
            with dpg.plot(label='Forces', height=200, width=900):
                # values = self._data[name]

                # optionally create legend
                dpg.add_plot_legend()

                x_axis_name = f'x_axis'
                y_axis_name = f'y_axis'

                # REQUIRED: create x and y axes
                dpg.add_plot_axis(dpg.mvXAxis, label="x", tag=x_axis_name)
                # dpg.set_axis_limits(x_axis_name, -n_points, 0)
                dpg.add_plot_axis(dpg.mvYAxis, label="y", tag=y_axis_name)
                # dpg.set_axis_limits(y_axis_name, -1, 1)
                
                for name, label in self._sorted_items:

                    # series belong to a y axis.
                    dpg.add_line_series(self._x_coords, [0] * len(self._x_coords), label=label, parent=y_axis_name, tag=name)

    
    def add_value(self, name, value):
        # return
        if name not in self._data:
            raise KeyError(f'Unknown data series {name}')
        
        self._data[name].append(value)
        if len(self._data[name]) > len(self._x_coords):
            self._data[name] = self._data[name][-len(self._x_coords):]        
        dpg.set_value(name, [self._x_coords[-len(self._data[name]):], self._data[name]])

