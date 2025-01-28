import customtkinter as ctk
from tkinter import filedialog, Canvas

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from panels import *


class ImageLoader(ctk.CTkFrame):
    def __init__(self, parent, load_func):
        super().__init__(master=parent, fg_color='transparent')
        self.grid(column=0, columnspan=2, row=0, sticky='nsew')
        self.load_func = load_func
        ctk.CTkLabel(self, text='Autonomes Fahren mit KI',
                     font=ctk.CTkFont(size=50, weight="bold")).pack(expand=True)
        ctk.CTkButton(self, text='Select image folder', font=ctk.CTkFont(size=20, weight="bold"), command=self.open_dialog, width=280, height=140).pack(expand=True)

    def open_dialog(self):
        path = filedialog.askdirectory()
        self.load_func(path)


class AugmentationSelector(ctk.CTkScrollableFrame):
    def __init__(self, parent, h_translate, v_translate, saturation_factor, switch_event, preview_augmentation):
        super().__init__(master=parent, fg_color='transparent')
        self.grid(column=0, row=0, sticky='nsew')

        ctk.CTkLabel(self, text='Augmentations', font=ctk.CTkFont(size=20, weight="bold"), pady=5).pack()
        self.panels = [MirroringPanel(self, switch_event, preview_augmentation),
                       TranslatePanel(self, h_translate, v_translate, switch_event, preview_augmentation),
                       ShadowPanel(self, switch_event, preview_augmentation),
                       ContrastPanel(self, switch_event, preview_augmentation),
                       GrayscalePanel(self, switch_event, preview_augmentation),
                       InvertColorPanel(self, switch_event, preview_augmentation),
                       RandomRectPanel(self, switch_event, preview_augmentation),
                       SaturationPanel(self, saturation_factor, switch_event, preview_augmentation),
                       EdgeDetectionPanel(self, switch_event, preview_augmentation)]


class TrainingOptions(ctk.CTkFrame):
    def __init__(self, parent, num_frames, apply_augmentations, train_model, available_models, start_driving):
        super().__init__(master=parent, fg_color='transparent')
        self.grid(column=1, row=0, sticky='nsew')

        ctk.CTkLabel(self, text='Training Options', font=ctk.CTkFont(size=20, weight="bold"), pady=5).pack()

        self.options_panel = TrainingOptionsPanel(self, num_frames)
        self.options_panel.pack(ipady=80)

        self.option_buttons = ctk.CTkFrame(self, fg_color='transparent')
        self.option_buttons.pack(fill='both', pady=50, ipady=20)

        self.option_buttons.columnconfigure((0, 1, 2), weight=1)
        self.option_buttons.rowconfigure((0, 1, 2), weight=1)

        self.augmentation_button = ctk.CTkButton(self.option_buttons, text='Augment Dataset', font=ctk.CTkFont(size=20, weight="bold"), width=200, height=100, command=apply_augmentations)
        self.augmentation_button.grid(column=0, row=0)
        self.training_button = ctk.CTkButton(self.option_buttons, text='Train Model', font=ctk.CTkFont(size=20, weight="bold"), width=200, height=100, command=train_model)
        self.training_button.grid(column=1, row=0)
        self.driving_button = ctk.CTkButton(self.option_buttons, text='Start Driving', font=ctk.CTkFont(size=20, weight="bold"), width=200, height=100, command=start_driving)
        self.driving_button.grid(column=2, row=0)
        self.available_models = ctk.CTkComboBox(self.option_buttons, values=available_models, state='readonly')
        self.available_models.set('Select Model')
        self.available_models.grid(column=2, row=1)
        self.augment_progress_bar = ctk.CTkProgressBar(self.option_buttons, orientation='horizontal', height=20, corner_radius=0)
        self.augment_progress_bar.set(value=0)
        self.augmenting_text = ctk.CTkLabel(self.option_buttons, text='Augmenting images...', font=ctk.CTkFont(weight="bold"))


class PreviewWindow(ctk.CTkToplevel):
    def __init__(self, parent, resize_preview_image):
        super().__init__(master=parent)
        self.title('Augmentation Preview')
        self.resizable(width=False, height=False)

        ctk.CTkLabel(self, text='Augmentation Preview', font=ctk.CTkFont(size=20, weight="bold"), pady=5).pack()
        ctk.CTkLabel(self, text='Original image:', font=ctk.CTkFont(size=15, weight="bold")).pack()
        self.original_canvas = PreviewCanvas(self, resize_preview_image)
        ctk.CTkLabel(self, text='Augmented image:', font=ctk.CTkFont(size=15, weight="bold")).pack()
        self.preview_canvas = PreviewCanvas(self, resize_preview_image)


class PreviewCanvas(Canvas):
    def __init__(self, parent, resize_preview_image):
        super().__init__(master=parent, background='#242424', bd=0, highlightthickness=0, relief='ridge')
        self.pack()

        self.bind('<Configure>', resize_preview_image)


class HistoryWindow(ctk.CTkToplevel):
    def __init__(self, parent):
        super().__init__(master=parent)
        self.geometry('800x500')
        self.title('Learning Progress')
        self.resizable(width=False, height=False)

        ctk.CTkLabel(self, text='Learning Progress', font=ctk.CTkFont(size=20, weight="bold"), pady=5).pack()

        self.training_progress_bar = ctk.CTkProgressBar(self, orientation='horizontal', height=20,
                                                       corner_radius=0, width=400)
        self.training_progress_bar.set(value=0)
        self.training_progress_bar.pack()

        self.fig = plt.figure()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=10)

    def update_plot(self, history, cur_epoch, epochs):
        loss = [d['loss'][0] for d in history]
        val_loss = [d['val_loss'][0] for d in history]
        x = range(1, cur_epoch + 1)
        plt.clf()
        plt.xlim(0, epochs + 1)
        plt.xticks(range(1, epochs + 1))
        plt.plot(x, loss, label="Loss")
        # plt.plot(cur_epoch, loss[-1], 'o')
        # plt.text(cur_epoch, loss[-1], f'{round(loss[-1], 4)}', fontsize=12, ha='center')
        plt.plot(x, val_loss, label="Validation Loss")
        # plt.plot(cur_epoch, val_loss[-1], 'o')
        # plt.text(cur_epoch, val_loss[-1], f'{round(val_loss[-1], 4)}', fontsize=12, ha='center')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        self.fig.legend(loc='upper right')
        self.canvas.draw()
