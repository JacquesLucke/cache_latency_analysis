import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

with open("C:\\Users\\jacques\\Documents\\performance_tests\\benchmark_result_save.csv") as f:
    loaded_results = f.read()

data = defaultdict(list)

fig = go.Figure()

data_by_name = defaultdict(lambda: ([], []))

for line in loaded_results.splitlines():
    name, x_value, y_value = line.split(";")
    data_by_name[name][0].append(int(x_value) * 64)
    data_by_name[name][1].append(float(y_value))

for name, (x_values, y_values) in data_by_name.items():
    fig.add_trace(go.Scatter(name=name, x=x_values, y=y_values))

fig.update_layout(
    barmode="group",
    xaxis_title="Total Bytes",
    yaxis_title="Time in Nanoseconds per Load",
    #xaxis_type="category",
    xaxis_tickformat=",.",
    legend_orientation="h",
    legend_x=0,
    legend_y=1.15,
    margin_t=0,
    showlegend=False,
    xaxis_type="log",
    xaxis_nticks=5)
#fig.update_yaxes(range=[0, 7], dtick=1)
#fig.write_image("/home/jacques/Documents/performance_tests/bit_iteration.png", width=800, height=350)
fig.show()
