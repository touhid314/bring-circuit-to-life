'''
the gui for running the program
'''

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage
from PyQt5.QtCore import Qt, QRect
from PyQt5 import QtCore
import numpy as np

from simulate import simulate_from_img


class ImageLabel(QLabel): 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_pixmap = None
        self.overlay_rect = QRect()
        self.component_name = ""
        self.voltage_rect = QRect()
        self.voltage_text = ""

    def setPixmap(self, pixmap):
        self.original_pixmap = pixmap
        super().setPixmap(pixmap)

    def setOverlay(self, overlay_rect, component_name="", voltage_rect=None, voltage_text=""):
        self.overlay_rect = overlay_rect
        self.component_name = component_name
        self.voltage_rect = voltage_rect if voltage_rect else QRect()
        self.voltage_text = voltage_text
        self.repaint()  # Trigger a repaint to draw the overlay


    def paintEvent(self, event):
        super().paintEvent(event)
        
        # Draw overlay and component name if overlay_rect is not null
        if not self.overlay_rect.isNull() and self.original_pixmap:
            painter = QPainter(self)
            painter.setBrush(QColor(0, 0, 200, 50))  # Semi-transparent blue
            painter.setPen(Qt.NoPen)
            painter.drawRect(self.overlay_rect)  # Draw the overlay

            # Draw component name above the overlay rectangle
            if self.component_name:
                painter.setPen(Qt.black)
                painter.drawText(self.overlay_rect.bottomLeft() - QtCore.QPoint(0, 0), self.component_name)

            # Draw voltage rectangle and text
            if not self.voltage_rect.isNull():
                painter.setBrush(QColor(0, 0, 255, 100))  # Semi-transparent blue
                painter.setPen(Qt.NoPen)
                painter.drawRect(self.voltage_rect)  # Draw the voltage rectangle
                
                # Draw the voltage text inside the rectangle
                painter.setPen(Qt.white)
                painter.drawText(self.voltage_rect, Qt.AlignCenter, f"V across: {self.voltage_text}")

            painter.end()


class ImageMouseTrackerApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle("Bring Circuit to Life!")
        self.setGeometry(100, 100, 600, 600)
        self.setMouseTracking(True)
        
        # Create a layout to organize widgets
        self.layout = QVBoxLayout(self)

        # Button to upload image
        self.upload_button = QPushButton("Upload Circuit Schematic Image", self)
        self.upload_button.clicked.connect(self.load_image)

        # Label to display the image with fixed size
        self.image_label = ImageLabel(self)
        self.image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.image_label.setFixedSize(500, 400)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setMouseTracking(True)

        # Label to display coordinates
        self.coord_label = QLabel("Click 'Upload Image' to load an image", self)

        # Add widgets to the layout
        self.layout.addWidget(self.upload_button)
        self.layout.addWidget(self.image_label)
        self.layout.addWidget(self.coord_label)

        self.original_pixmap = None
        self.scaled_pixmap_rect = QRect()

        
        # Component mapping and bounding boxes
        self.component_mapping = {0: 'capacitor_unpolarized', 1: 'inductor', 2: 'resistor', 3: 'vdc'}
        self.elec_comp_bbox = []
        self.comp_voltages = []
        

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
                   


        ################################################################
        # SIMULATION AND EVERYTHING HERE
        self.coord_label.setText("BRINGING LIFE TO CIRCUIT....")

        # simulate for the given image path
        # print(file_path)      
        

        if file_path:
            self.elec_comp_bbox, self.comp_voltages, NODE_MAP, combined_img = simulate_from_img(file_path)
            
            # Convert PIL Image to QPixmap
            combined_img_qt = combined_img.convert("RGBA")
            data = combined_img_qt.tobytes("raw", "RGBA")
            qimage = QImage(data, combined_img_qt.width, combined_img_qt.height, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)
            
            self.original_pixmap = pixmap
            scaled_pixmap = self.original_pixmap.scaled(self.image_label.width(), self.image_label.height(),
                                                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.scaled_pixmap_rect = QRect(0, 0, scaled_pixmap.width(), scaled_pixmap.height())
        else:
            print("no file path was specified")


        self.coord_label.setText("BROUGHT LIFE! HOVER TO FIND!")



    def mouseMoveEvent(self, event):
        if self.image_label.pixmap():
            x = event.pos().x() - self.image_label.x()
            y = event.pos().y() - self.image_label.y()

            if self.scaled_pixmap_rect.contains(x, y):
                orig_x = int(x * self.original_pixmap.width() / self.scaled_pixmap_rect.width())
                orig_y = int(y * self.original_pixmap.height() / self.scaled_pixmap_rect.height())
                self.coord_label.setText(f"Image pixel coordinates: (x: {orig_x}, y: {orig_y})")

                # Iterate through component bounding boxes 
                for comp, voltage in zip(self.elec_comp_bbox, self.comp_voltages):
                    if(len(comp) == 6):
                        comp_class, comp_x, comp_y, comp_w, comp_h, _ = comp
                    else:
                        comp_class, comp_x, comp_y, comp_w, comp_h= comp
                    comp_name = self.component_mapping.get(comp_class, "Unknown")
                    voltage_value = voltage[1]  # Voltage value for the component
                    
                    # Calculate top-left and bottom-right of bbox for current component
                    comp_x1 = int((comp_x - comp_w / 2) * self.original_pixmap.width())
                    comp_y1 = int((comp_y - comp_h / 2) * self.original_pixmap.height())
                    comp_x2 = int((comp_x + comp_w / 2) * self.original_pixmap.width())
                    comp_y2 = int((comp_y + comp_h / 2) * self.original_pixmap.height())

                    # Check if mouse is within bbox
                    if comp_x1 <= orig_x <= comp_x2 and comp_y1 <= orig_y <= comp_y2:
                        overlay_x = int(comp_x1 * self.scaled_pixmap_rect.width() / self.original_pixmap.width())
                        overlay_y = int(comp_y1 * self.scaled_pixmap_rect.height() / self.original_pixmap.height())
                        overlay_w = int(comp_w * self.scaled_pixmap_rect.width())
                        overlay_h = int(comp_h * self.scaled_pixmap_rect.height())
                        overlay_rect = QRect(overlay_x, overlay_y, overlay_w, overlay_h)

                        # Create a blue rectangle for voltage display
                        voltage_rect_x = overlay_x
                        voltage_rect_y = overlay_y - 30  # Position it slightly above the component box
                        voltage_rect_w = 80  # Width of the voltage display box
                        voltage_rect_h = 20  # Height of the voltage display box
                        voltage_rect = QRect(voltage_rect_x, voltage_rect_y, voltage_rect_w, voltage_rect_h)

                        # Create voltage text
                        voltage_text = f"{voltage_value:.2f}V"

                        # Update the overlay and voltage display
                        self.image_label.setOverlay(overlay_rect, component_name=comp_name, 
                                                    voltage_rect=voltage_rect, voltage_text=voltage_text)
                        return

                self.image_label.setOverlay(QRect())  # Clear overlay if outside any bbox
            else:
                self.coord_label.setText("Mouse outside image bounds")
                self.image_label.setOverlay(QRect())  # Clear overlay if outside range

# Run the app
app = QApplication(sys.argv)
window = ImageMouseTrackerApp()
window.show()
sys.exit(app.exec_())
