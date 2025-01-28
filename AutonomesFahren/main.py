import os
import re
from threading import Thread

import numpy as np
from PIL import ImageTk, Image

import behavioral_cloning_drive
import behavioral_cloning_train
from behavioral_cloning_train import *
from app_widgets import *
from augment_data import *


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode('dark')
        self.geometry('1200x500')
        self.title('Autonomes Fahren mit KI')
        self.resizable(width=False, height=False)
        self.init_parameters()

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=2, uniform='a')
        self.columnconfigure(1, weight=5, uniform='a')

        self.image_loader = ImageLoader(self, self.image_loader)

        self.protocol('WM_DELETE_WINDOW', self.Exit)

        self.mainloop()

    def init_parameters(self):
        self.h_translate = ctk.IntVar(value=25)
        self.v_translate = ctk.IntVar(value=5)
        self.saturation_factor = ctk.DoubleVar(value=1.0)
        self.enable_count = 0
        self.preview_window = None
        self.preview_image = None
        self.models = None
        self.is_driving = False
        self.history_window = None
        self.frame_count = 0
        self.augmented_images_count = 0
        self.train_history = []

    def image_loader(self, path):
        self.image_path = path

        if self.image_path != "":
            self.load_images()

            self.image_loader.grid_forget()
            self.augmentation_selector = AugmentationSelector(self, self.h_translate, self.v_translate, self.saturation_factor, self.switch_event, self.preview_augmentation)
            self.training_options = TrainingOptions(self, self.frame_count, self.apply_augmentations, self.train_model, self.models, self.start_driving)
            self.load_models()

    def switch_event(self):
        self.enable_count = 0

        for p in self.augmentation_selector.panels:
            if p.enable_switch.get():
                self.enable_count += 1

        if self.enable_count == 3:
            for p in self.augmentation_selector.panels:
                if not p.enable_switch.get():
                    p.enable_switch.configure(state='disabled')
        else:
            for p in self.augmentation_selector.panels:
                if not p.enable_switch.get():
                    p.enable_switch.configure(state='normal')

    def load_images(self):
        self.frame_count = len(next(os.walk(self.image_path))[2])
        self.frame_center_count = sum([len([file for file in files if ('center' in file)]) for root, dirs, files in sorted(os.walk(self.image_path))])

        img, steer = utils.load_image(self.image_path, os.listdir(self.image_path)[0], include_left_right=False, include_augmented=False)

        if img is None:
            return

        self.preview_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB')

    def preview_augmentation(self, augmentation):
        augmented_img, steer, augmentation_name = self.get_augmented_image(augmentation, cv2.cvtColor(np.array(self.preview_image),
                                                                                   cv2.COLOR_RGB2BGR), 0)
        self.preview_augmented_image_tk = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(augmented_img, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB'))

        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.preview_window = PreviewWindow(self, self.resize_preview_image)
            self.preview_image_tk = ImageTk.PhotoImage(self.preview_image)

            self.preview_window.after(100, self.preview_window.lift)
        else:
            self.preview_window.preview_canvas.itemconfig(self.preview_canvas, image=self.preview_augmented_image_tk)
            self.preview_window.focus()

    def resize_preview_image(self, event):
        self.preview_window.preview_canvas.delete('all')
        self.preview_canvas = self.preview_window.preview_canvas.create_image(event.width / 2, event.height / 2, image=self.preview_augmented_image_tk)
        self.preview_window.original_canvas.delete('all')
        self.preview_window.original_canvas.create_image(event.width / 2, event.height / 2, image=self.preview_image_tk)

    def validate_model_name(self, model_name):
        if not bool(re.match(r"^[A-Za-z0-9_-]*$", model_name)) or model_name == '':
            self.training_options.options_panel.model_name.configure(fg_color='red2')
            self.update()
            return False
        else:
            self.model_name = model_name
            self.training_options.options_panel.model_name.configure(fg_color='#343638')
            self.update()
            return True

    def apply_augmentations(self):
        if self.enable_count == 0:
            return

        self.set_button_states('disabled')

        self.training_options.augment_progress_bar.grid(column=0, row=2)
        self.training_options.augmenting_text.configure(text='Augmenting images...', text_color='#DCE4EE')
        self.training_options.augmenting_text.grid(column=0, row=1)

        for file in os.listdir(self.image_path):
            if not file.endswith(".png"):
                continue

            img, steer = utils.load_image(self.image_path, file, include_left_right=False, include_augmented=False)

            if img is None:
                continue

            for p in self.augmentation_selector.panels:
                temp_img = img
                temp_steer = steer
                if p.enable_switch.get():
                    temp_img, temp_steer, augmentation_name = self.get_augmented_image(p, temp_img, temp_steer)

                    self.augmented_images_count += 1
                    self.training_options.augment_progress_bar.set(value=self.augmented_images_count/(self.frame_center_count*self.enable_count))
                    self.update()

                    filename = f"{file[0:5]}_s{temp_steer:+.3f}_augment_{augmentation_name}"
                    filename = filename.replace(".", ",")
                    filename += ".png"

                    cv2.imwrite(os.path.join(self.image_path, filename), temp_img)

        self.load_images()
        self.training_options.options_panel.num_frames.configure(text=f'Dataset size: {self.frame_count}')
        self.training_options.augmenting_text.configure(text='Augmentations finished!', text_color='green2')
        self.update()
        self.training_options.augment_progress_bar.grid_forget()
        self.training_options.augment_progress_bar.set(value=0)
        self.augmented_images_count = 0
        self.set_button_states('normal')

    def get_augmented_image(self, augmentation, img, steer):
        augmentation_name = None

        if augmentation.enable_switch.cget('text') == 'Mirroring':
            if augmentation.enable_h.get():
                img, steer = utils.flip_horizontal(img, steer)
                augmentation_name = 'horizontal'
            if augmentation.enable_v.get():
                img, steer = utils.flip_vertical(img, steer)
                augmentation_name = 'vertical'
        elif augmentation.enable_switch.cget('text') == 'Translate':
            img, steer = utils.random_translate(img, steer, augmentation.h_translate_slider.get(), augmentation.v_translate_slider.get())
            augmentation_name = augmentation.enable_switch.get()
        elif augmentation.enable_switch.cget('text') == 'Shadow':
            img = utils.random_shadow(img)
            augmentation_name = augmentation.enable_switch.get()
        elif augmentation.enable_switch.cget('text') == 'Maximum Contrast':
            img = utils.maximum_contrast(img)
            augmentation_name = augmentation.enable_switch.get()
        elif augmentation.enable_switch.cget('text') == 'Grayscale':
            img = utils.to_grayscale(img)
            augmentation_name = augmentation.enable_switch.get()
        elif augmentation.enable_switch.cget('text') == 'Invert Color':
            img = utils.invert_color(img)
            augmentation_name = augmentation.enable_switch.get()
        elif augmentation.enable_switch.cget('text') == 'Rectangle':
            img = utils.add_random_rectangle(img)
            augmentation_name = augmentation.enable_switch.get()
        elif augmentation.enable_switch.cget('text') == 'Saturation':
            img = utils.change_saturation(img, saturation_factor=augmentation.saturation_factor_slider.get())
            augmentation_name = augmentation.enable_switch.get()
        elif augmentation.enable_switch.cget('text') == 'Edge Detection':
            img = utils.edge_detection(img)
            augmentation_name = augmentation.enable_switch.get()

        return img, steer, augmentation_name

    def train_model(self):
        if self.validate_model_name(self.training_options.options_panel.model_name.get()):
            self.num_epochs = int(self.training_options.options_panel.num_epochs.get())
            self.batch_size = int(self.training_options.options_panel.batch_size.get())
            self.set_button_states('disabled')

            if self.history_window is None or not self.history_window.winfo_exists():
                self.history_window = HistoryWindow(self)
                self.history_window.after(100, self.history_window.lift)
            else:
                self.history_window.focus()

            self.update()

            self.fit_model()
            self.load_models()

            self.set_button_states('normal')

    def load_models(self):
        self.models = os.listdir('./models/')
        self.training_options.available_models.configure(values=self.models)

    def start_driving(self):
        if not self.is_driving:
            self.selected_model = self.training_options.available_models.get()
            if self.selected_model != 'Model name':
                self.is_driving = True
                self.driving_thread = Thread(target=behavioral_cloning_drive.start_autonomous_driving, args=(self.selected_model, lambda: self.is_driving))
                self.driving_thread.start()
                self.training_options.driving_button.configure(text='Stop Driving', fg_color='red2')
        else:
            self.training_options.driving_button.configure(text='Start Driving', fg_color='#1F6AA5')
            self.is_driving = False
            self.driving_thread.join()

    def fit_model(self):
        model = behavioral_cloning_train.build_model()

        (imgs, steers) = behavioral_cloning_train.load_data(self.image_path)

        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1.0e-4))

        for i in range(self.num_epochs):
            print(f'Epoch {i + 1}/{self.num_epochs}')
            model.fit(imgs, steers, epochs=1, batch_size=self.batch_size, validation_split=0.2)
            self.train_history.append(model.history.history)
            self.history_window.update_plot(self.train_history, cur_epoch=i+1, epochs=self.num_epochs)
            self.history_window.training_progress_bar.set(value=(i+1)/self.num_epochs)
            self.update()

        self.history_window.training_progress_bar.configure(progress_color='green2')
        model.save(filepath=f"models/{self.model_name}.keras", overwrite=True)

    def set_button_states(self, state):
        self.training_options.training_button.configure(state=state)
        self.training_options.augmentation_button.configure(state=state)
        self.training_options.driving_button.configure(state=state)

    def Exit(self):
        plt.close()
        self.quit()


App()
