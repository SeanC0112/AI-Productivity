from PyQt5 import QtWidgets, QtCore, QtGui
import sys

app = QtWidgets.QApplication(sys.argv)

label = QtWidgets.QLabel()
# Remove window borders and set it to stay on top
label.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.WindowStaysOnTopHint)
# Allow the background to be transparent
label.setAttribute(QtCore.Qt.WA_TranslucentBackground)

pixmap = QtGui.QPixmap("Cat/Idle/tile000.png")
label.setPixmap(pixmap)
label.move(0,-10)
label.show()

timer = QtCore.QTimer()
timer.timeout.connect(lambda: label.setPixmap(QtGui.QPixmap("Cat/Idle/tile001.png")))  # Update the image every second
timer.start(3000)  # Change the image every 1000 milliseconds (1 second

timer2 = QtCore.QTimer()
timer2.timeout.connect(lambda: label.hide())  # Update the image every second
timer2.start(7000)  #


sys.exit(app.exec())
