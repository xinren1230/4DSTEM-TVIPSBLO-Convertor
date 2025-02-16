#!/usr/bin/env python3
import sys, os, time, math
import numpy as np
import h5py
import cv2
import subprocess
from scipy.signal import find_peaks
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from PyQt5 import uic
from PyQt5.QtWidgets import (QApplication, QWizard, QWizardPage, QFileDialog,
                             QMessageBox, QLabel, QVBoxLayout, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSlot, QPoint, pyqtSignal, QProcess, QEvent
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------- Helper Functions ----------------
def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def load_image(frame, hf_path, digit, scale):
    with h5py.File(hf_path, 'r') as hf:
        group = hf.get('Individual_Images')
        file_name = 'frame_' + str(frame).zfill(digit)
        n1 = group.get(file_name)
        if n1 is None:
            raise ValueError(f"Dataset '{file_name}' not found in HDF5 file: {hf_path}")
        image = np.array(n1)
    return resize_image(image, scale)

def load_images(numframes, hf_path, digit, scale):
    images = []
    with h5py.File(hf_path, 'r') as hf:
        group = hf.get('Individual_Images')
        for i in range(numframes):
            file_name = 'frame_' + str(i).zfill(digit)
            n1 = group.get(file_name)
            if n1 is None:
                continue
            image = np.array(n1)
            images.append(resize_image(image, scale))
    return images

def convert_np_to_pixmap(np_img):
    if np_img.dtype != np.uint8:
        np_img = np_img.astype('float32')
        np_img = 255 * (np_img - np_img.min()) / (np_img.max() - np_img.min() + 1e-5)
        np_img = np_img.astype('uint8')
    if len(np_img.shape) == 2:
        height, width = np_img.shape
        bytes_per_line = width
        q_img = QImage(np_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
    elif len(np_img.shape) == 3:
        height, width, channels = np_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(np_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
    else:
        return None
    return QPixmap.fromImage(q_img)

def virtual_bf_mask(image, centeroffsetpx=(0.0,), radiuspx=10):
    xx, yy = np.meshgrid(np.arange(image.shape[0], dtype=float),
                         np.arange(image.shape[1], dtype=float))
    xx -= 0.5 * image.shape[0] + centeroffsetpx[0]
    yy -= 0.5 * image.shape[1] + centeroffsetpx[0]
    mask = np.hypot(xx, yy) < radiuspx
    return mask

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB):
    m = mse(imageA, imageB)
    s = ssim(imageA, imageB)
    return m, s

def add_mask(mask, frame):
    return np.multiply(frame, np.invert(mask))

def find_start_from_vbf(vbf_image):
    last_row = vbf_image[-1, :]
    peaks, _ = find_peaks(-last_row, prominence=0)
    new_guess = None
    for idx, val in enumerate(last_row):
        if val < 230:
            new_guess = idx
            break
    return new_guess

# ---------------- Custom Widget: DraggableMaskLabel ----------------
class DraggableMaskLabel(QLabel):
    maskMoved = pyqtSignal(tuple)
    offsetUpdated = pyqtSignal(int, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.mask_radius = 10
        self.mask_center = None
        self.dragging = False
        self.drag_offset = QPoint()
        self.setMouseTracking(True)

    def setImage(self, image, mask_radius=10):
        self.image = image
        self.mask_radius = mask_radius
        h, w = image.shape[:2]
        self.mask_center = QPoint(w // 2, h // 2)
        self.updateOverlay()

    def updateOverlay(self):
        if self.image is None:
            return
        pixmap = convert_np_to_pixmap(self.image)
        overlay = QPixmap(pixmap)
        painter = QPainter(overlay)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(255, 0, 0))
        painter.setBrush(QColor(255, 0, 0, 128))
        painter.drawEllipse(self.mask_center, self.mask_radius, self.mask_radius)
        painter.end()
        self.setPixmap(overlay)
        if self.mask_center:
            self.maskMoved.emit((self.mask_center.x(), self.mask_center.y()))

    def mousePressEvent(self, event):
        if self.mask_center is None:
            return super().mousePressEvent(event)
        if (event.pos() - self.mask_center).manhattanLength() <= self.mask_radius:
            self.dragging = True
            self.drag_offset = event.pos() - self.mask_center
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.dragging:
            new_center = event.pos() - self.drag_offset
            self.mask_center = new_center
            self.updateOverlay()
            if self.image is not None:
                h, w = self.image.shape[:2]
                offset_x = new_center.x() - (w // 2)
                offset_y = new_center.y() - (h // 2)
                self.offsetUpdated.emit(offset_x, offset_y)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.dragging = False
        super().mouseReleaseEvent(event)

    def getMaskCenter(self):
        if self.mask_center:
            return (self.mask_center.x(), self.mask_center.y())
        return None

# ---------------- Page 1: Parameter Selection & Mask Check ----------------
class ParameterAndMaskCheckPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("ParameterAndMaskCheckPage.ui", self)
        # Register fields (ensure UI object names match these)
        self.registerField("vbfH", self.vbfHSpin)
        self.registerField("vbfW", self.vbfWSpin)
        self.registerField("tvipsFile*", self.tvipsFileLineEdit)
        self.registerField("linescale", self.linescaleEdit)
        self.registerField("numframes", self.numframesSpin)
        self.registerField("whichFrame", self.whichFrameSpin)
        self.registerField("offsetX", self.offsetXSpin)
        self.registerField("offsetY", self.offsetYSpin)
        self.registerField("radius", self.radiusSpin)
        self.registerField("scale", self.scaleSpin)
        self.registerField("guessFirstFrame", self.guessFirstFrameSpin)

        self.browseButton.clicked.connect(self.browseFile)
        self.checkMaskButton.clicked.connect(self.checkMaskPosition)

        self.offsetXSpin.setMinimum(-1000)
        self.offsetYSpin.setMinimum(-1000)

        self.offsetXSpin.valueChanged.connect(self.updateMaskFromFields)
        self.offsetYSpin.valueChanged.connect(self.updateMaskFromFields)
        self.radiusSpin.valueChanged.connect(self.updateMaskFromFields)

        self.draggableMask = DraggableMaskLabel(self)
        layout = self.maskCheckGroup.layout()
        layout.replaceWidget(self.maskImageLabel, self.draggableMask)
        self.maskImageLabel.deleteLater()
        self.draggableMask.offsetUpdated.connect(self.updateOffsetSpinBoxes)
        self.draggableMask.maskMoved.connect(self.updateMaskPositionLabel)

        # Flag to indicate whether the mask has been checked.
        self.maskChecked = False

    def initializePage(self):
        # Set default Virtual BF dimensions and reset the mask flag.
        self.vbfHSpin.setValue(10)
        self.vbfWSpin.setValue(256)
        self.maskChecked = False

    def isComplete(self):
        # This page is complete only if the mask has been checked.
        return self.maskChecked

    @pyqtSlot()
    def browseFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Select TVIPS File", "", "TVIPS Files (*.tvips)")
        if fname:
            self.tvipsFileLineEdit.setText(fname)
            self.guessFirstFrameSpin.setValue(1)
        
        
    @pyqtSlot()
    def checkMaskPosition(self):
        file_path = self.tvipsFileLineEdit.text().strip()
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select TVIPS File", "", "TVIPS Files (*.tvips)")
            if not file_path:
                return
            self.tvipsFileLineEdit.setText(file_path)
        self.wizard().setProperty("tvipsFile", file_path)
        file_name = os.path.basename(file_path)
        folder_path = os.path.dirname(file_path)
        file_name_hf = file_name[:-10] + ".h5"
        hf_path = os.path.join(folder_path, file_name_hf)
        self.wizard().setProperty("hfPath", hf_path)

        linescale = self.field("linescale")
        numframes = self.field("whichFrame") + 1
        run_string = (
            'python ./tvips/recorderR.py --otype=Individual --linscale=' + str(linescale) +
            ' --numframes=' + str(numframes) +
            ' "' + file_path + '" "' + hf_path + '"'
        )
        print("Running command:", run_string)
        self.progressBarStep1.setRange(0, 0)
        self.proc1 = QProcess(self)
        self.proc1.setProcessChannelMode(QProcess.MergedChannels)
        self.proc1.finished.connect(lambda ec, es: self.loadImageAfterCommand(ec, es))
        self.proc1.start(run_string)

    @pyqtSlot(int, int)
    def loadImageAfterCommand(self, exitCode, exitStatus):
        print("Command finished with exitCode =", exitCode, "and exitStatus =", exitStatus)
        self.progressBarStep1.setRange(0, 100)
        self.progressBarStep1.setValue(100)
        whichFrame = self.field("whichFrame")
        digit = len(str(whichFrame - 1))
        hf_path = self.wizard().property("hfPath")
        try:
            image = load_image(int(whichFrame), hf_path, digit, 100)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading image: {e}")
            return
        radius = self.field("radius")
        self.draggableMask.setImage(image, mask_radius=radius)
        h, w = image.shape[:2]
        self.draggableMask.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.draggableMask.setFixedSize(w, h)
        self.draggableMask.adjustSize()
        self.maskCheckGroup.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.maskCheckGroup.adjustSize()
        self.adjustSize()
        self.wizard().adjustSize()
        pos = self.draggableMask.getMaskCenter()
        if pos:
            self.maskPositionLabel.setText(f"Mask Position: ({pos[0]}, {pos[1]})")
        # Mark the mask as checked and notify the wizard that the page is complete.
        self.maskChecked = True
        self.completeChanged.emit()

    @pyqtSlot()
    def updateMaskFromFields(self):
        if self.draggableMask.image is None:
            return
        h, w = self.draggableMask.image.shape[:2]
        new_center = QPoint(w // 2 + self.offsetXSpin.value(),
                            h // 2 + self.offsetYSpin.value())
        self.draggableMask.mask_center = new_center
        self.draggableMask.mask_radius = self.radiusSpin.value()
        self.draggableMask.updateOverlay()
        self.maskPositionLabel.setText(f"Mask Position: ({new_center.x()}, {new_center.y()})")

    @pyqtSlot(int, int)
    def updateOffsetSpinBoxes(self, x, y):
        self.offsetXSpin.setValue(x)
        self.offsetYSpin.setValue(y)

    @pyqtSlot(tuple)
    def updateMaskPositionLabel(self, pos):
        self.maskPositionLabel.setText(f"Mask Position: ({pos[0]}, {pos[1]})")


# ---------------- Page 2: VBF Generation & Analysis ----------------
class VBFMergedPage(QWizardPage):
    def __init__(self, parent=None):
        super(VBFMergedPage, self).__init__(parent)
        uic.loadUi("VBF_MergedPage.ui", self)
        self.processFinished = False

        # Set up the matplotlib canvas inside the plot container.
        self.figure = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figure)
        self.rerunButton.clicked.connect(self.rerunVBF)

        # Install event filter on the vbfImageLabel so that clicks update newGuessSpin.
        self.vbfImageLabel.installEventFilter(self)

        # Flag to ensure we connect the spin box signals only once.
        self._vbf_spin_connections_added = False

    def eventFilter(self, obj, event):
        if obj == self.vbfImageLabel and event.type() == QEvent.MouseButtonPress:
            pos = event.pos()
            # Retrieve the current starting frame; default to 0 if not set.
            first_frame = self.wizard().property("firstFrame") or 0
            # Retrieve the original virtual BF image width (vbfW parameter)
            vbf_w = int(self.field("vbfW") or 256)
            # Calculate the scaling factor between the original width and the label's width.
            scaling_factor = vbf_w / self.vbfImageLabel.width()
            # Update the starting frame based on the click position.
            first_frame = first_frame + int(pos.x() * scaling_factor)
            # Update the newGuessSpin control on the current page.
            self.newGuessSpin.setValue(first_frame)
            # Store the new starting frame in the wizard's property.
            self.wizard().setProperty("firstFrame", first_frame)
            # Update the guessFirstFrameSpin on the first page (assumed to be at index 0).
            first_page = self.wizard().page(0)
            if hasattr(first_page, "guessFirstFrameSpin"):
                first_page.guessFirstFrameSpin.setValue(first_frame)
            return True
        return super(VBFMergedPage, self).eventFilter(obj, event)

    def initializePage(self):
        scale_percentage = int(self.field("scale") or 20)
        offsetX = int(self.field("offsetX") or 4)
        offsetY = int(self.field("offsetY") or 4)
        radius = int(self.field("radius") or 10)
        linescale = self.field("linescale") or "0-2000"
        vbf_h = int(self.field("vbfH") or 10)
        vbf_w = int(self.field("vbfW") or 256)
        first_frame = self.wizard().property("firstFrame")
        if first_frame is None:
            first_frame = 0
        hf_path = self.wizard().property("hfPath")
        if not os.path.exists(hf_path):
            self.vbfOutputTextEdit.append("HDF5 file not found. Running individual image generation (step 2)...")
            linescale_ind = self.field("linescale")
            numframes_ind = self.field("whichFrame") + 1
            tvipsFile = self.wizard().property("tvipsFile")
            cmd = ('python ./tvips/recorderR.py --otype=Individual --linscale=' + str(linescale_ind) +
                   ' --numframes=' + str(numframes_ind) +
                   ' "' + tvipsFile + '" "' + hf_path + '"')
            proc = QProcess(self)
            proc.setProcessChannelMode(QProcess.MergedChannels)
            proc.start(cmd)
            proc.waitForFinished()

        # Synchronize the Virtual BF dimensions with the first page.
        first_page = self.wizard().page(0)
        if hasattr(first_page, "vbfHSpin") and hasattr(first_page, "vbfWSpin"):
            # Set initial values for this page's spin boxes based on the first page.
            self.vbfHSpin.setValue(first_page.vbfHSpin.value())
            self.vbfWSpin.setValue(first_page.vbfWSpin.value())
            # Connect changes on this page to update the first page (only once).
            if not self._vbf_spin_connections_added:
                self.vbfHSpin.valueChanged.connect(lambda val: first_page.vbfHSpin.setValue(val))
                self.vbfWSpin.valueChanged.connect(lambda val: first_page.vbfWSpin.setValue(val))
                self._vbf_spin_connections_added = True

        numframes_vbf = (vbf_h * vbf_w) + int(first_frame)
        self.vbfOutputTextEdit.clear()
        self.vbfOutputTextEdit.append("Running command to generate Virtual BF image...")
        tvipsFile = self.wizard().property("tvipsFile")
        command = (
            'python ./tvips/recorderR.py --otype=VirtualBF --linscale=' + str(linescale) +
            ' --numframes=' + str(numframes_vbf) +
            ' --vbfradius=' + str(radius) +
            ' --vbfcenter=' + str(offsetX) + 'x' + str(offsetY) +
            ' --dimension=' + str(vbf_h) + 'x' + str(vbf_w) +
            ' --skip=' + str(first_frame) +
            ' "' + tvipsFile + '" "' + hf_path + '"'
        )
        self.vbfOutputTextEdit.append("Command: " + command)
        if hasattr(self, 'vbfProgressBar'):
            self.vbfProgressBar.setRange(0, 0)
        self.vbfProcess = QProcess(self)
        self.vbfProcess.setProcessChannelMode(QProcess.MergedChannels)
        self.vbfProcess.readyReadStandardOutput.connect(self.handleVBFOutput)
        self.vbfProcess.finished.connect(self.virtualBFFinished)
        self.vbfProcess.start(command)

    def handleVBFOutput(self):
        data = self.vbfProcess.readAllStandardOutput().data().decode()
        self.vbfOutputTextEdit.append(data)

    def virtualBFFinished(self):
        self.vbfOutputTextEdit.append("Virtual BF generation finished.")
        if hasattr(self, 'vbfProgressBar'):
            self.vbfProgressBar.setRange(0, 100)
            self.vbfProgressBar.setValue(100)
        hf_path = self.wizard().property("hfPath")
        try:
            with h5py.File(hf_path, 'r') as hf:
                g1 = hf.get('Virtual_bright_field')
                n1 = g1.get('Virtual_bright_field')
                vbf = np.array(n1)
            pixmap = convert_np_to_pixmap(vbf)
            if pixmap:
                # Display the original size image in vbfImageLabel.
                self.vbfImageLabel.setPixmap(pixmap)
            else:
                self.vbfImageLabel.setText("Error converting image.")
        except Exception as e:
            self.vbfOutputTextEdit.append(f"Error loading Virtual BF image: {e}")
            return

    def rerunVBF(self):
        new_guess = self.newGuessSpin.value()
        tvipsFile = self.wizard().property("tvipsFile")
        hf_path = self.wizard().property("hfPath")
        linescale = self.field("linescale")
        radius = self.field("radius")
        offsetX = int(self.field("offsetX") or 4)
        offsetY = int(self.field("offsetY") or 4)
        vbf_h = int(self.field("vbfH") or 10)
        vbf_w = int(self.field("vbfW") or 256)
        numframes_vbf = (vbf_h * vbf_w) + new_guess
        command = (
            'python ./tvips/recorderR.py --otype=VirtualBF --linscale=' + str(linescale) +
            ' --numframes=' + str(numframes_vbf) +
            ' --vbfradius=' + str(radius) +
            ' --vbfcenter=' + str(offsetX) + 'x' + str(offsetY) +
            ' --dimension=' + str(vbf_h) + 'x' + str(vbf_w) +
            ' --skip=' + str(new_guess) +
            ' "' + tvipsFile + '" "' + hf_path + '"'
        )
        self.vbfOutputTextEdit.append("Running command: " + command)
        if hasattr(self, 'vbfProgressBar'):
            self.vbfProgressBar.setRange(0, 0)
        self.vbfProcess = QProcess(self)
        self.vbfProcess.setProcessChannelMode(QProcess.MergedChannels)
        self.vbfProcess.readyReadStandardOutput.connect(self.handleVBFOutput)
        self.vbfProcess.finished.connect(self.virtualBFFinished)
        self.vbfProcess.start(command)

# ---------------- Page 3: Conversion & Batch Conversion ----------------
class ConversionAndBatchPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        uic.loadUi("ConversionAndBatchPage.ui", self)
        self.addFileButton.clicked.connect(self.addFile)
        self.runConversionButton.clicked.connect(self.runBatchConversion)
        self.runSingleConversionButton.clicked.connect(self.runSingleConversion)
        self.addFileButton.clicked.connect(self.addFile)
        self.addTaskButton.clicked.connect(self.addtask)

        # This flag indicates whether the conversion process has completed.
        self._conversion_complete = False

        # Internal list to store file paths (HDF5 files)
        self.file_list = []

    def isComplete(self):
        # The Finish button will be enabled only if conversion is complete.
        return self._conversion_complete

    def addFile(self):
        # Open a file dialog to choose a file (e.g., a BLO file or another HDF5 file)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select File", "", "HDF5 Files (*.h5);;All Files (*.*)"
        )
        if file_path:

            # Add the file name (or full path) to the list widget for display
            self.fileListWidget.addItem(file_path)
            self.logTextEdit.append(f"Added file: {file_path}")

    def addtask(self):
        # Save conversion parameters to the HDF5 file specified in hf_path
        try:
            hf_path = self.wizard().property("hfPath")
            first_frame = self.wizard().property("firstFrame")
            vbf_w = self.wizard().property("vbf_w") or self.field("vbfW")
            linescale = self.wizard().property("linescale") or self.field("linescale")
            file_path = self.wizard().property("tvipsFile")
            diff_image = self.wizard().property("diff_image")
            diff_size = diff_image.shape[1] if diff_image is not None else -1
            blo_h = self.bloHSpin.value()

            with h5py.File(hf_path, 'r+') as hf:
                g1_name = 'Parameters_for_conversion'
                if g1_name in hf:
                    del hf[g1_name]
                g1 = hf.create_group(g1_name)
                g1.create_dataset('Starting_frame', data=first_frame)
                g1.create_dataset('Image_height', data=blo_h)
                g1.create_dataset('Image_width', data=vbf_w)
                # Corrected dataset name from 'Image_constrast' to 'Image_contrast'
                g1.create_dataset('Image_contrast', data=linescale)
                g1.create_dataset('File_path', data=file_path)
                g1.create_dataset('Diff_size', data=diff_size)

            self.logTextEdit.append("Conversion parameters saved successfully.")
            # Add the HDF5 file path to our file list (if desired)
            self.logTextEdit.append(f"Added file: {hf_path}")
            self.fileListWidget.addItem(hf_path)
        except Exception as e:
            self.logTextEdit.append(f"Error saving conversion parameters: {e}")

            
    def runSingleConversion(self):
        # Run conversion for a single file (hf_path from wizard properties)
        try:
            hf_path = self.wizard().property("hfPath")
            first_frame = self.wizard().property("firstFrame")
            vbf_w = self.wizard().property("vbf_w") or self.field("vbfW")
            linescale = self.wizard().property("linescale") or self.field("linescale")
            file_path = self.wizard().property("tvipsFile")
            diff_image = self.wizard().property("diff_image")
            diff_size = diff_image.shape[1] if diff_image is not None else -1
            blo_h = self.bloHSpin.value()
            with h5py.File(hf_path, 'r+') as hf:
                g1_name = 'Parameters_for_conversion'
                if g1_name in hf:
                    del hf[g1_name]
                g1 = hf.create_group(g1_name)
                g1.create_dataset('Starting_frame', data=first_frame)
                g1.create_dataset('Image_height', data=blo_h)
                g1.create_dataset('Image_width', data=vbf_w)
                g1.create_dataset('Image_constrast', data=linescale)
                g1.create_dataset('File_path', data=file_path)
                g1.create_dataset('Diff_size', data=diff_size)
            self.logTextEdit.append("Conversion parameters saved successfully.")
        except Exception as e:
            self.logTextEdit.append(f"Error saving conversion parameters: {e}")
            return

        try:
            use_filter = self.useFilterCheck.isChecked()
            median = self.medianSpin.value()
            gaussian = self.gaussianLineEdit.text()
            binning = self.binningSpin.value()
            hf_path = self.wizard().property("hfPath")
            if not hf_path:
                self.logTextEdit.append("Conversion file not found.")
                return
            run_string_0 = 'python ./tvips/recorderR.py --otype=blo'
            filter_string = (f" --median={median} --gaussian={gaussian}") if use_filter else ''
            binning_string = '' if binning == 1 else f" --binning={binning}"
            
            with h5py.File(hf_path, 'r') as hf:
                g1_name = 'Parameters_for_conversion'
                n1 = hf.get(g1_name)
                linescale_dataset = n1.get('Image_constrast')[()]
                linescale_string = ' --linscale=' + str(linescale_dataset)[1:].replace("'", "")
                image_string = ' --dimension=' + str(n1.get('Image_height')[()]) + 'x' + str(n1.get('Image_width')[()])
                skip_string = ' --skip=' + str(n1.get('Starting_frame')[()])
                file_path_string = str(n1.get('File_path')[()])[1:].replace("'", "")
                diff_size_value = n1.get('Diff_size')[()]
                diff_size_value = int(diff_size_value/binning)
            if use_filter:
                blo_path_string = file_path_string[:-10] + '_' + str(diff_size_value) + 'F.blo'
            else:
                blo_path_string = file_path_string[:-10] + '_' + str(diff_size_value) + '.blo'
            
            run_string = (run_string_0 + linescale_string + binning_string + image_string +
                          skip_string + filter_string + ' "' + file_path_string + '" ' +
                          '"' + blo_path_string + '"')
            self.logTextEdit.append("Running: " + run_string)
            self.progressBar.setRange(0, 0)
            self.process = QProcess(self)
            self.process.setProcessChannelMode(QProcess.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.handleOutput)
            self.process.readyReadStandardError.connect(self.handleOutput)
            self.process.finished.connect(self.conversionFinished)
            self.process.start(run_string)
        except Exception as e:
            self.logTextEdit.append(f"Error processing file {hf_path}: {e}")

    def runBatchConversion(self):
        try:
            use_filter = self.useFilterCheck.isChecked()
            median = self.medianSpin.value()
            gaussian = self.gaussianLineEdit.text()
            binning = self.binningSpin.value()
            run_string_0 = 'python ./tvips/recorderR.py --otype=blo'
            filter_string = f" --median={median} --gaussian={gaussian}" if use_filter else ''
            binning_string = '' if binning == 1 else f" --binning={binning}"

            # Update the internal file list from fileListWidget
            self.file_list = []
            for index in range(self.fileListWidget.count()):
                item = self.fileListWidget.item(index)
                self.file_list.append(item.text())

            # Loop over each file in the internal file list
            for file in self.file_list:
                try:
                    with h5py.File(file, 'r') as hf:
                        g1_name = 'Parameters_for_conversion'
                        n1 = hf.get(g1_name)
                        # Assuming the dataset name is "Image_contrast"
                        linescale_dataset = n1.get('Image_contrast')[()]
                        # Convert the dataset value to a string for the command-line parameter.
                        linescale_string = ' --linscale=' + str(linescale_dataset)[1:].replace("'", "")
                        image_string = ' --dimension=' + str(n1.get('Image_height')[()]) + 'x' + str(n1.get('Image_width')[()])
                        skip_string = ' --skip=' + str(n1.get('Starting_frame')[()])
                        file_path_string = str(n1.get('File_path')[()])[1:].replace("'", "")
                        diff_size_value = n1.get('Diff_size')[()]
                        diff_size_value = int(diff_size_value / binning)
                    if use_filter:
                        blo_path_string = file_path_string[:-10] + '_' + str(diff_size_value) + 'F.blo'
                    else:
                        blo_path_string = file_path_string[:-10] + '_' + str(diff_size_value) + '.blo'

                    run_string = (run_string_0 + linescale_string + binning_string + image_string +
                                  skip_string + filter_string + ' "' + file_path_string + '" ' +
                                  '"' + blo_path_string + '"')
                    self.logTextEdit.append("Running: " + run_string)
                    self.progressBar.setRange(0, 0)
                    self.process = QProcess(self)
                    self.process.setProcessChannelMode(QProcess.MergedChannels)
                    self.process.readyReadStandardOutput.connect(self.handleOutput)
                    self.process.readyReadStandardError.connect(self.handleOutput)
                    self.process.finished.connect(self.conversionFinished)
                    self.process.start(run_string)
                except Exception as e:
                    self.logTextEdit.append(f"Error processing file {file}: {e}")
        except Exception as e:
            self.logTextEdit.append(f"Error in runBatchConversion: {e}")


    def handleOutput(self):
        data = self.process.readAllStandardOutput().data().decode()
        self.logTextEdit.append(data)

    def conversionFinished(self):
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(100)
        self.logTextEdit.append("Batch conversion completed!")
        QMessageBox.information(self, "Batch Conversion", "Batch conversion completed successfully!")
        # Mark conversion as complete and notify the wizard.
        self._conversion_complete = True
        self.completeChanged.emit()



# ---------------- Main Wizard ----------------
class MyWizard(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("4DSTEM TVIPS to BLO convertor")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinimizeButtonHint)
        self.addPage(ParameterAndMaskCheckPage())
        self.addPage(VBFMergedPage())
        self.addPage(ConversionAndBatchPage())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    wizard = MyWizard()
    wizard.show()
    sys.exit(app.exec_())
