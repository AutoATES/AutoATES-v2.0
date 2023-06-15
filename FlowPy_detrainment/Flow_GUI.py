# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\Flow_GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.5
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(546, 352)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(".\\FlowPylogo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.forest_Label = QtWidgets.QLabel(self.centralwidget)
        self.forest_Label.setObjectName("forest_Label")
        self.gridLayout.addWidget(self.forest_Label, 6, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 12, 1, 1, 1)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 4, 1, 1, 1)
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.gridLayout.addWidget(self.line_5, 8, 3, 1, 1)
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout.addWidget(self.line_4, 4, 3, 1, 1)
        self.Release_Button = QtWidgets.QToolButton(self.centralwidget)
        self.Release_Button.setObjectName("Release_Button")
        self.gridLayout.addWidget(self.Release_Button, 3, 4, 1, 1)
        self.calc_Button = QtWidgets.QPushButton(self.centralwidget)
        self.calc_Button.setObjectName("calc_Button")
        self.gridLayout.addWidget(self.calc_Button, 13, 3, 1, 1)
        self.infra_Button = QtWidgets.QToolButton(self.centralwidget)
        self.infra_Button.setObjectName("infra_Button")
        self.gridLayout.addWidget(self.infra_Button, 5, 4, 1, 1)
        self.line_6 = QtWidgets.QFrame(self.centralwidget)
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.gridLayout.addWidget(self.line_6, 11, 3, 1, 1)
        self.infra_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.infra_lineEdit.setObjectName("infra_lineEdit")
        self.gridLayout.addWidget(self.infra_lineEdit, 5, 3, 1, 1)
        self.wDir_Button = QtWidgets.QToolButton(self.centralwidget)
        self.wDir_Button.setCheckable(False)
        self.wDir_Button.setObjectName("wDir_Button")
        self.gridLayout.addWidget(self.wDir_Button, 1, 4, 1, 1)
        self.release_label = QtWidgets.QLabel(self.centralwidget)
        self.release_label.setObjectName("release_label")
        self.gridLayout.addWidget(self.release_label, 3, 1, 1, 1)
        self.wDir_label = QtWidgets.QLabel(self.centralwidget)
        self.wDir_label.setObjectName("wDir_label")
        self.gridLayout.addWidget(self.wDir_label, 1, 1, 1, 1)
        self.DEM_Button = QtWidgets.QToolButton(self.centralwidget)
        self.DEM_Button.setObjectName("DEM_Button")
        self.gridLayout.addWidget(self.DEM_Button, 2, 4, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout.addWidget(self.label_7)
        self.alpha_Edit = QtWidgets.QLineEdit(self.centralwidget)
        self.alpha_Edit.setEnabled(True)
        self.alpha_Edit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.alpha_Edit.setReadOnly(False)
        self.alpha_Edit.setClearButtonEnabled(False)
        self.alpha_Edit.setObjectName("alpha_Edit")
        self.horizontalLayout.addWidget(self.alpha_Edit)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout.addWidget(self.label_6)
        self.exp_Edit = QtWidgets.QLineEdit(self.centralwidget)
        self.exp_Edit.setEnabled(True)
        self.exp_Edit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.exp_Edit.setReadOnly(False)
        self.exp_Edit.setObjectName("exp_Edit")
        self.horizontalLayout.addWidget(self.exp_Edit)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.gridLayout.addLayout(self.horizontalLayout, 9, 3, 1, 1)
        self.DEM_label = QtWidgets.QLabel(self.centralwidget)
        self.DEM_label.setObjectName("DEM_label")
        self.gridLayout.addWidget(self.DEM_label, 2, 1, 1, 1)
        self.DEM_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.DEM_lineEdit.setObjectName("DEM_lineEdit")
        self.gridLayout.addWidget(self.DEM_lineEdit, 2, 3, 1, 1)
        self.infra_label = QtWidgets.QLabel(self.centralwidget)
        self.infra_label.setObjectName("infra_label")
        self.gridLayout.addWidget(self.infra_label, 5, 1, 1, 1)
        self.wDir_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.wDir_lineEdit.setObjectName("wDir_lineEdit")
        self.gridLayout.addWidget(self.wDir_lineEdit, 1, 3, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout.addWidget(self.line_3, 11, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 9, 1, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout.addWidget(self.line_2, 8, 1, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.flux_Edit = QtWidgets.QLineEdit(self.centralwidget)
        self.flux_Edit.setObjectName("flux_Edit")
        self.horizontalLayout_3.addWidget(self.flux_Edit)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.z_Edit = QtWidgets.QLineEdit(self.centralwidget)
        self.z_Edit.setObjectName("z_Edit")
        self.horizontalLayout_3.addWidget(self.z_Edit)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.gridLayout.addLayout(self.horizontalLayout_3, 10, 3, 1, 1)
        self.release_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.release_lineEdit.setObjectName("release_lineEdit")
        self.gridLayout.addWidget(self.release_lineEdit, 3, 3, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.outputBox = QtWidgets.QComboBox(self.centralwidget)
        self.outputBox.setObjectName("outputBox")
        self.outputBox.addItem("")
        self.outputBox.addItem("")
        self.horizontalLayout_2.addWidget(self.outputBox)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem4)
        self.gridLayout.addLayout(self.horizontalLayout_2, 12, 3, 1, 1)
        self.forest_lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.forest_lineEdit.setObjectName("forest_lineEdit")
        self.gridLayout.addWidget(self.forest_lineEdit, 6, 3, 1, 1)
        self.forest_Button = QtWidgets.QToolButton(self.centralwidget)
        self.forest_Button.setObjectName("forest_Button")
        self.gridLayout.addWidget(self.forest_Button, 6, 4, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 546, 21))
        self.menubar.setDefaultUp(False)
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionLoad = QtWidgets.QAction(MainWindow)
        self.actionLoad.setObjectName("actionLoad")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionLoad)
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Flow-Py"))
        self.forest_Label.setText(_translate("MainWindow", "Forest"))
        self.label.setText(_translate("MainWindow", "Output Format"))
        self.Release_Button.setText(_translate("MainWindow", "..."))
        self.calc_Button.setText(_translate("MainWindow", "Calculate"))
        self.infra_Button.setText(_translate("MainWindow", "..."))
        self.wDir_Button.setText(_translate("MainWindow", "..."))
        self.release_label.setText(_translate("MainWindow", "Release Layer"))
        self.wDir_label.setText(_translate("MainWindow", "Working Directory"))
        self.DEM_Button.setText(_translate("MainWindow", "..."))
        self.label_7.setText(_translate("MainWindow", "alpha"))
        self.label_6.setText(_translate("MainWindow", "exp    "))
        self.DEM_label.setText(_translate("MainWindow", "DEM Layer"))
        self.infra_label.setText(_translate("MainWindow", "Infrastructure"))
        self.label_2.setText(_translate("MainWindow", "Parameters"))
        self.label_4.setText(_translate("MainWindow", "flux   "))
        self.label_3.setText(_translate("MainWindow", "max_z"))
        self.outputBox.setItemText(0, _translate("MainWindow", ".tif"))
        self.outputBox.setItemText(1, _translate("MainWindow", ".asc"))
        self.forest_Button.setText(_translate("MainWindow", "..."))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionLoad.setText(_translate("MainWindow", "Load"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
