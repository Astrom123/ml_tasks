from PyQt5.QtWidgets import QApplication, QPushButton, QGridLayout, QWidget
from PyQt5.QtWidgets import QLineEdit, QLabel
from humor_classifier import HumorClassifier
import sys


class MainWindow(QWidget):

    def __init__(self):
        self.humor_clf = HumorClassifier()
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Humor Classifier")
        self.resize(500, 100)

        grid = QGridLayout()

        self.textbox = QLineEdit(self)
        self.textbox.setPlaceholderText('Введите текст')
        grid.addWidget(self.textbox, 0, 0)

        self.res_textbox = QLabel(self)
        grid.addWidget(self.res_textbox, 1, 0)

        self.check_button = QPushButton("Check", self)
        self.check_button.clicked.connect(self.is_humorous)
        grid.addWidget(self.check_button, 0, 2)

        self.setLayout(grid)

    def is_humorous(self):
        if self.humor_clf.is_humorous(self.textbox.text()):
            self.res_textbox.setText("Смешно :D")
        else:
            self.res_textbox.setText("Не смешно :/")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
