import plotly.graph_objects as go
import plotly.express as px
from collections import defaultdict

with open("/home/jacques/Documents/performance_tests/benchmark_results.csv") as f:
    loaded_results = f.read()

data = defaultdict(list)

fig = go.Figure()

data_by_function = defaultdict(list)

for line in loaded_results.splitlines():
    parts = line.split(";")
    data_by_function[parts[1]].append(line)

for function, lines in data_by_function.items():
    x_values = []
    y_values = []
    for line in lines:
        parts = line.split(";")
        x_values.append(f"{int(parts[0]):,}")
        y_values.append(parts[-1])
    fig.add_trace(go.Bar(x=x_values, y=y_values, name=function))

fig.update_layout(
    barmode="group",
    xaxis_title="Number of Non-Zeros (of 10,000,000)",
    yaxis_title="Time in Milliseconds",
    xaxis_type="category",
    xaxis_tickformat=",.",
    legend_orientation="h",
    legend_x=0,
    legend_y=1.2,
    margin_t=0,
    margin_r=10)
fig.write_image("/home/jacques/Documents/performance_tests/counting_bits.png", width=800, height=350)
fig.show()
