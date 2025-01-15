from os.path import abspath, dirname

import folium

map_file_path = abspath(dirname(dirname(__file__)))
output_figure_map_path = map_file_path + '/output_figures'

# Sample data
data = [
    {"name": "Lugar 1", "lat": 19.4326, "lon": 1.1332},
    {"name": "Lugar 2", "lat": 34.0522, "lon": 0.2437},
    {"name": "Lugar 3", "lat": 40.7128, "lon": -3.0060},
]


def print_map(data, output_file_path=output_figure_map_path + '/map.html'):
    map_center = [40.4165, -3.70256]  # Center in Madrid
    m = folium.Map(location=map_center, zoom_start=6, prefer_canvas=True)

    for idx, row in data.iterrows():
        folium.CircleMarker(location=[row['Lat'], row['Lon']], radius=2, weight=5).add_to(m)

    m.save(output_file_path)


if __name__ == '__main__':
    print_map(data)
