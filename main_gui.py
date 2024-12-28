'''
the gui for running the program
'''

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QFileDialog, QVBoxLayout, QPushButton, QHBoxLayout, QScrollArea, QTextEdit
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage, QFontDatabase, QFont
from PyQt5.QtCore import Qt, QRect, QThread
from PyQt5 import QtCore
import numpy as np

from simulate import simulate_from_img
from llm_ui import process_prompt

class ImageLabel(QLabel): 
    '''
    class to show interactive overlays on top of the image 
    '''
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

    def setOverlay(self, overlay_rect, component_name="", voltage_rect=None, voltage_text="", element_name=""):
        self.overlay_rect = overlay_rect
        self.component_name = f"{component_name}, {element_name}"
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
                painter.setPen(Qt.white)
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



class PlotThread(QThread):
    def __init__(self, plt, axe):
        super().__init__()
        self.plt = plt
        self.axe = axe

    def run(self):
        show_plot(self.plt, self.axe)

def show_plot(plt, axe):
    import mplcursors
    from PyQt5.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox

    plt.grid(True)
    axe.axhline(0, color='black', linewidth=1, linestyle='--')
    axe.axvline(0, color='black', linewidth=1, linestyle='--')
    
    # Dynamically set xlim and ylim
    x_min, x_max = axe.get_xlim()
    y_min, y_max = axe.get_ylim()
    
    axe.set_xlim(left=max(x_min, -15), right=min(x_max, 15))
    axe.set_ylim(bottom=max(y_min, -15), top=min(y_max, 15))
    
    mplcursors.cursor(axe, hover=True)
    plt.gcf().set_size_inches(10, 5)  # Change the figure size

    # Create a new dialog window
    dialog = QDialog()
    dialog.setWindowTitle("Plot")
    dialog.setMinimumSize(800, 600)

    # Create a layout for the dialog
    layout = QVBoxLayout(dialog)

    # Create a canvas for the plot
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    canvas = FigureCanvas(plt.gcf())
    layout.addWidget(canvas)

    # Add a button box for closing the dialog
    button_box = QDialogButtonBox(QDialogButtonBox.Ok)
    button_box.accepted.connect(dialog.accept)
    layout.addWidget(button_box)

    dialog.setLayout(layout)
    dialog.exec_()
    plt.clf()  # Clear the current figure



class ImageMouseTrackerApp(QWidget):
    def __init__(self):
        super().__init__()

        ############################ APP INTERFACE ###################################
        # Load the Roboto font
        QFontDatabase.addApplicationFont("./font/Roboto-Medium.ttf")
        self.setFont(QFont("Roboto"))

        # Set up the window
        self.setWindowTitle("Bring Circuit to Life!")
        self.setGeometry(100, 100, 1000, 500)  # Adjusted width to accommodate both areas
        self.setMouseTracking(True)
        
        # Main horizontal layout
        self.main_layout = QHBoxLayout(self)

        # Left vertical layout
        self.left_layout = QVBoxLayout()
        self.main_layout.addLayout(self.left_layout)

        # Right vertical layout
        self.right_layout = QVBoxLayout()
        self.main_layout.addLayout(self.right_layout)

        # Button to upload image
        self.upload_button = QPushButton("Upload Circuit Schematic Image", self)
        self.upload_button.setFixedWidth(500)
        self.upload_button.clicked.connect(self.load_image)

        # Label to display the image with fixed size and placeholder image
        self.image_label = ImageLabel(self)
        self.image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.image_label.setFixedSize(500, 500)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setMouseTracking(True)
        self.image_label.setStyleSheet("background-color: #222; border: 0px solid #444;")

        # Set a placeholder image
        placeholder_pixmap = QPixmap(500, 500)
        placeholder_pixmap.fill(QColor("#333"))
        self.image_label.setPixmap(placeholder_pixmap)

        # Label to display coordinates
        self.coord_label = QLabel("Click 'Upload Image' to load an image", self)
        self.coord_label.setStyleSheet("color: #ddd;")

        # Text area for user input
        self.textbox = QTextEdit(self)
        self.textbox.setFixedSize(450, 80)
        self.textbox.setPlaceholderText("Talk with the circuit")
        self.textbox.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.textbox.setStyleSheet("background-color: #333; color: #ddd; padding: 10px; border: 1px solid #444;")
        self.textbox.installEventFilter(self)


        # Button to ask circuit
        self.ask_button = QPushButton("Ask Circuit", self)
        self.ask_button.setFixedSize(100, 30)  # Smaller width and height
        self.ask_button.setStyleSheet("margin-top: 10px; background-color: #555; color: #ddd; border: 1px solid #444;")
        self.ask_button.clicked.connect(self.ask_circuit)

        # Text area for displaying long strings and paragraphs
        self.display_area = QLabel(self)
        self.display_area.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.display_area.setWordWrap(True)
        self.set_chat_area_text("Hello! Upload a circuit schematic image to start simulating!")
        self.display_area.setStyleSheet("border: 0px solid black; padding: 10px;")
        self.display_area.setTextInteractionFlags(Qt.TextSelectableByMouse)  # Make text selectable and copyable
        self.display_area.adjustSize()  # Adjust size to fit the content
        self.textbox.textChanged.connect(self.scroll_to_bottom) # TODO: 


        # Scroll area to contain the display area
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedSize(450, 400)
        self.scroll_area.setWidget(self.display_area)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
            background-color: #222;
            border: 1px solid #444;
            }
            QScrollBar:vertical {
            background-color: #333;
            width: 12px;
            margin: 0px 3px 0px 3px;
            }
            QScrollBar::handle:vertical {
            background-color: #555;
            min-height: 20px;
            border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            background: none;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: none;
            }
        """)


        # Add the scroll area to the right layout
        self.right_layout.addWidget(self.scroll_area)

        # Add widgets to the left layout
        self.left_layout.addWidget(self.upload_button)
        self.left_layout.addWidget(self.image_label)
        self.left_layout.addWidget(self.coord_label)

        # Add widgets to the right layout
        self.right_layout.addWidget(self.textbox)
        self.right_layout.addWidget(self.ask_button, alignment=Qt.AlignRight)  # Align button to the right
        # self.setStyleSheet("background-color: #222; color: #ddd;")

         # Set the overall style for the main window
        self.setStyleSheet("""
            QWidget {
                background-color: #222;
                color: #ddd;
                font-family: Roboto;
                font-size: 12px;
            }
            QLabel, QLineEdit, QPushButton, QScrollArea {
                font-family: Roboto;
                font-size: 12px;
            }
            ImageLabel{
                font-size: 10px;
                color: white;
            }
        """)

        self.original_pixmap = None
        self.scaled_pixmap_rect = QRect()

        ################################### REQUIRED PROPERTIES #####################
        # Component mapping and bounding boxes
        self.component_mapping = {0: 'Capacitor(Unpolarized)', 1: 'Inductor', 2: 'Resistor', 3: 'VDC'}
        self.elec_comp_bbox = []
        self.comp_voltages = []
    
        # necessary variables
        self.ckt_netlist= None
        self.analyzer = None
        self.llm_model_path = "llm\models\llama-2-13b-chat.ggmlv3.q5_1.bin"
        
        #loads the llm model at the start of the program
        from llm.llm_model import get_llm_model
        bypass = True 
        
        if(bypass):
            self.llm_model = None
        else:
            self.llm_model = get_llm_model(model_path=self.llm_model_path, show_execution_time=True)



    def scroll_to_bottom(self):
        # Ensure the scrollbar is at the bottom by default
        self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())

    def set_chat_area_text(self, text, append=True):
        if append:
            current_text = self.display_area.text()
            new_text = current_text + "\n" + text
            self.display_area.setText(new_text)
        else:
            self.display_area.setText(text)

    def eventFilter(self, source, event):
        """
        for detection of enter key pressed in the prompt input
        """
        if event.type() == QtCore.QEvent.KeyPress and source is self.textbox:
            if event.key() == Qt.Key_Return and not (event.modifiers() & Qt.ShiftModifier):
                self.ask_circuit()
                return True
        return super().eventFilter(source, event)



    def ask_circuit(self, prompt=None):

        if(not prompt):
            user_input = self.textbox.toPlainText()
        else:
            user_input = prompt

        if user_input:
            if(prompt):
                show_text = f"\n\n►" # TODO: change variable name to text_to_disp 

            else:
                show_text = f"\n\n► {user_input}"

            self.set_chat_area_text(show_text)

            ########## INTEGRATE LLM HERE #########
            response = process_prompt(prompt=user_input, llm_model=self.llm_model, ckt_netlist=self.ckt_netlist, analyzer=self.analyzer)
            
            # QApplication.processEvents()  
            # QtCore.QThread.sleep(1)  

            print(response)
            if response['plt']!=None:
                show_plot(response['plt'], response['axe'])
            
            response = response['non_exec_response']

            #######################################
            
            show_text =  "\n••• "+ response

            self.set_chat_area_text(show_text)
            self.textbox.clear()  # Clear the text area after processing the question
        else:
            self.set_chat_area_text("\n••• Please enter a question.")

    def load_image(self):
        """
        load an image and perform netlist creation and initial simulation
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.jpeg *.bmp)")
                   
        # simulate for the given image path
        # print(file_path)      
    
        if file_path:
            
            self.coord_label.setText("BRINGING LIFE TO CIRCUIT....") # TODO: why does not this work?
            # self.set_chat_area_text("BRINGING LIFE TO CIRCUIT....")

            ################## SIMULATING #############################
            # simulate
            self.elec_comp_bbox, self.comp_voltages, combined_img, self.ckt_netlist, self.analyzer = simulate_from_img(file_path)
            self.set_chat_area_text("Brought circuit to life!",append=False)
            # send greeting as a circuit
            self.ask_circuit("hello, greet me as a circuit ")
            ###########################################################

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
            self.coord_label.setText("BROUGHT LIFE! HOVER TO FIND!")

        else:
            print("no file path was specified")
            self.coord_label.setText("no file path was specified")

    def mouseMoveEvent(self, event):
        """
        interactively show components information in the image area
        """
        if self.image_label.pixmap():
            x = event.pos().x() - self.image_label.x()
            y = event.pos().y() - self.image_label.y()

            if self.scaled_pixmap_rect.contains(x, y):
                orig_x = int(x * self.original_pixmap.width() / self.scaled_pixmap_rect.width())
                orig_y = int(y * self.original_pixmap.height() / self.scaled_pixmap_rect.height())
                self.coord_label.setText(f"Image pixel coordinates: (x: {orig_x}, y: {orig_y})")

                # Iterate through component bounding boxes 
                for comp, voltage in zip(self.elec_comp_bbox, self.comp_voltages):
                    # if(len(comp) == 6):
                    comp_class, comp_x, comp_y, comp_w, comp_h, orientation, el_name = comp
                    # else:
                    #     comp_class, comp_x, comp_y, comp_w, comp_h= comp
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
                                                    voltage_rect=voltage_rect, voltage_text=voltage_text, element_name = el_name)
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
