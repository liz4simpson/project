import flet as ft
import joblib as jb
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import matplotlib
import matplotlib.pyplot as plt
from flet.matplotlib_chart import MatplotlibChart

matplotlib.use("svg")

model = "small.joblib"

class PageWidget:
    def __init__(self, page: ft.Page):
        self._page = page
        self._filepath_text = ft.Text(value="Selected file path")
        self._filepicker = ft.FilePicker(on_result=self._return_file)
        self._filepicker_row = self._create_filepicker_row()
        self._predict_text = ft.Text()
        self._predict_btn = ft.ElevatedButton(
            text="Просчитать", on_click=self._button_clicked
        )

        page.title = "SalesVision"

    def render(self):
        self._page.horizontal_alignment = "center"
        self._page.vertical_alignment = "top"
        self._page.padding = 25
        self._page.theme_mode = "light"

        self._page.add(
            self._filepicker_row,
            ft.Divider(thickness=1),
            ft.Divider(thickness=1),
            self._predict_btn,
            self._predict_text,
        )

        self._page.update()

    def _create_filepicker_row(self):
        row = ft.Row()
        row.controls.append(
            ft.ElevatedButton(
                text="Выберите данные...", on_click=self._select_file
            )
        )
        row.controls.append(self._filepath_text)

        return row

    def _select_file(self, e):
        self._page.add(self._filepicker)
        self._filepicker.pick_files("Select file...")

    def _return_file(self, e: ft.FilePickerResultEvent):
        filepath = e.files[0].path
        # Заменяем обратные слеши на прямые
        filepath = filepath.replace("\\", "/")

        self._filepath_text.value = filepath
        self._page.update()

    def _button_clicked(self, e):

        kmeans = jb.load(model)
        data = pd.read_csv(self._filepath_text.value)  # Замените "your_dataset.csv" на путь к вашему файлу с данными
        X = data[['Open', 'Close', 'High', 'Low']]
        kmeans.predict(X)
        # Добавление меток кластеров к исходным данным
        data['Cluster'] = kmeans.labels_

        # Визуализация кластеров
        print("Start")
        fig, ax = plt.subplots()
        plt.scatter(data['Date'], data['Close'], c=data['Cluster'], cmap='viridis')
        ax.set_xlabel('Date')
        ax.set_ylabel('Close Price')
        ax.set_title('Clusters of Stock Prices')
        self._file_preview = MatplotlibChart(fig, expand=True)
        self._page.add(MatplotlibChart(fig, expand=True))
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        self._predict_text.value = f"Средний индекс силуэта: {silhouette_avg}"

        self._page.update()
        print("Stop")


def main(page: ft.Page):
    page_widget = PageWidget(page)
    page_widget.render()


ft.app(target=main)
