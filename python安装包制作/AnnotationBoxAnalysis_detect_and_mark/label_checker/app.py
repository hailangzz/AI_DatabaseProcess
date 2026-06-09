# label_checker/app.py

import sys
from PyQt5.QtWidgets import QApplication
from .gui import LabelChecker

def main():
    app = QApplication(sys.argv)
    win = LabelChecker()
    win.show()
    sys.exit(app.exec_())