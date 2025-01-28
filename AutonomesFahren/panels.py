import customtkinter as ctk


class Panel(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(master=parent, fg_color='gray')
        self.pack(fill='both', pady=4, ipady=20, padx=4)


class MirroringPanel(Panel):
    def __init__(self, parent, switch_event, preview_augmentation):
        super().__init__(parent=parent)

        self.rowconfigure((0, 1), weight=1)
        self.columnconfigure((0, 1, 2), weight=1)

        self.enable_switch = ctk.CTkSwitch(self, text='Mirroring', command=switch_event, onvalue='mirroring')
        self.enable_switch.grid(column=0, row=0, sticky='w', padx=10)

        ctk.CTkButton(self, text='Preview', width=20, command=lambda p=self: preview_augmentation(p)).grid(column=2, row=0, sticky='e', padx=10)

        self.enable_v = ctk.CTkCheckBox(self, text='Vertical')
        self.enable_v.grid(column=0, row=1)

        self.enable_h = ctk.CTkCheckBox(self, text='Horizontal')
        self.enable_h.grid(column=2, row=1)


class TranslatePanel(Panel):
    def __init__(self, parent, h_translate, v_translate, switch_event, preview_augmentation):
        super().__init__(parent=parent)

        self.rowconfigure((0, 1, 2, 3, 4), weight=1)
        self.columnconfigure((0, 1, 2), weight=1)

        self.enable_switch = ctk.CTkSwitch(self, text='Translate', command=switch_event, onvalue='translate')
        self.enable_switch.grid(column=0, row=0, sticky='w', padx=10)

        ctk.CTkButton(self, text='Preview', width=20, command=lambda: preview_augmentation(self)).grid(column=2, row=0, sticky='e', padx=10)

        ctk.CTkLabel(self, text='Vertical').grid(column=0, row=1, sticky='w', padx=10)
        self.v_translate_slider = ctk.CTkSlider(self, from_=0, to=10, variable=v_translate)
        self.v_translate_slider.grid(column=0, row=2, pady=2)
        ctk.CTkLabel(self, textvariable=v_translate).grid(column=1, row=1, sticky='e')

        ctk.CTkLabel(self, text='Horizontal').grid(column=0, row=3, sticky='w', padx=10)
        self.h_translate_slider = ctk.CTkSlider(self, from_=0, to=50, variable=h_translate)
        self.h_translate_slider.grid(column=0, row=4, pady=2)
        ctk.CTkLabel(self, textvariable=h_translate).grid(column=1, row=3, sticky='e')


class ShadowPanel(Panel):
    def __init__(self, parent, switch_event, preview_augmentation):
        super().__init__(parent=parent)

        self.rowconfigure((0), weight=1)
        self.columnconfigure((0, 1, 2), weight=1)

        self.enable_switch = ctk.CTkSwitch(self, text='Shadow', command=switch_event, onvalue='shadow')
        self.enable_switch.grid(column=0, row=0, sticky='w', padx=10)

        ctk.CTkButton(self, text='Preview', width=20, command=lambda: preview_augmentation(self)).grid(column=2, row=0, sticky='e', padx=10)


class ContrastPanel(Panel):
    def __init__(self, parent, switch_event, preview_augmentation):
        super().__init__(parent=parent)

        self.rowconfigure((0), weight=1)
        self.columnconfigure((0, 1, 2), weight=1)

        self.enable_switch = ctk.CTkSwitch(self, text='Maximum Contrast', command=switch_event, onvalue='maxContrast')
        self.enable_switch.grid(column=0, row=0, sticky='w', padx=10)

        ctk.CTkButton(self, text='Preview', width=20, command=lambda: preview_augmentation(self)).grid(column=2, row=0, sticky='e', padx=10)


class GrayscalePanel(Panel):
    def __init__(self, parent, switch_event, preview_augmentation):
        super().__init__(parent=parent)

        self.rowconfigure((0), weight=1)
        self.columnconfigure((0, 1, 2), weight=1)

        self.enable_switch = ctk.CTkSwitch(self, text='Grayscale', command=switch_event, onvalue='grayscale')
        self.enable_switch.grid(column=0, row=0, sticky='w', padx=10)

        ctk.CTkButton(self, text='Preview', width=20, command=lambda: preview_augmentation(self)).grid(column=2, row=0, sticky='e', padx=10)


class InvertColorPanel(Panel):
    def __init__(self, parent, switch_event, preview_augmentation):
        super().__init__(parent=parent)

        self.rowconfigure((0), weight=1)
        self.columnconfigure((0, 1, 2), weight=1)

        self.enable_switch = ctk.CTkSwitch(self, text='Invert Color', command=switch_event, onvalue='invertColor')
        self.enable_switch.grid(column=0, row=0, sticky='w', padx=10)

        ctk.CTkButton(self, text='Preview', width=20, command=lambda: preview_augmentation(self)).grid(column=2, row=0, sticky='e', padx=10)


class RandomRectPanel(Panel):
    def __init__(self, parent, switch_event, preview_augmentation):
        super().__init__(parent=parent)

        self.rowconfigure((0), weight=1)
        self.columnconfigure((0, 1, 2), weight=1)

        self.enable_switch = ctk.CTkSwitch(self, text='Rectangle', command=switch_event, onvalue='rectangle')
        self.enable_switch.grid(column=0, row=0, sticky='w', padx=10)

        ctk.CTkButton(self, text='Preview', width=20, command=lambda: preview_augmentation(self)).grid(column=2, row=0, sticky='e', padx=10)


class SaturationPanel(Panel):
    def __init__(self, parent, saturation_factor, switch_event, preview_augmentation):
        super().__init__(parent=parent)

        self.rowconfigure((0, 1, 2), weight=1)
        self.columnconfigure((0, 1, 2), weight=1)

        self.enable_switch = ctk.CTkSwitch(self, text='Saturation', command=switch_event, onvalue='saturation')
        self.enable_switch.grid(column=0, row=0, sticky='w', padx=10)

        ctk.CTkButton(self, text='Preview', width=20, command=lambda: preview_augmentation(self)).grid(column=2, row=0, sticky='e', padx=10)

        ctk.CTkLabel(self, text='Saturation factor').grid(column=0, row=1, sticky='w', padx=10)
        self.saturation_factor_slider = ctk.CTkSlider(self, from_=0, to=2, variable=saturation_factor, number_of_steps=100, command=self.update_text)
        self.saturation_factor_slider.grid(column=0, row=2, pady=2)
        self.saturation_factor_label = ctk.CTkLabel(self, text=saturation_factor.get())
        self.saturation_factor_label.grid(column=1, row=1, sticky='e')

    def update_text(self, value):
        self.saturation_factor_label.configure(text=f'{round(value, 2)}')


class EdgeDetectionPanel(Panel):
    def __init__(self, parent, switch_event, preview_augmentation):
        super().__init__(parent=parent)

        self.rowconfigure((0), weight=1)
        self.columnconfigure((0, 1, 2), weight=1)

        self.enable_switch = ctk.CTkSwitch(self, text='Edge Detection', command=switch_event, onvalue='edgeDetection')
        self.enable_switch.grid(column=0, row=0, sticky='w', padx=10)

        ctk.CTkButton(self, text='Preview', width=20, command=lambda: preview_augmentation(self)).grid(column=2, row=0, sticky='e', padx=10)


class TrainingOptionsPanel(Panel):
    def __init__(self, parent, num_frames):
        super().__init__(parent=parent)

        self.rowconfigure((0, 1, 2), weight=1)
        self.columnconfigure((0, 1, 2, 3), weight=1)

        self.num_frames = ctk.CTkLabel(self, text=f'Dataset size: {num_frames}', font=ctk.CTkFont(size=20, weight="bold"))
        self.num_frames.grid(row=0, column=0, sticky='e')

        ctk.CTkLabel(self, text='Model name: ', font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=2, sticky='e')
        self.model_name = ctk.CTkEntry(self, placeholder_text='Model name')
        self.model_name.grid(row=0, column=3, sticky='w')

        ctk.CTkLabel(self, text='Number of epochs: ', font=ctk.CTkFont(size=20, weight="bold")).grid(row=1, column=0, sticky='e')
        self.num_epochs = ctk.CTkComboBox(self, values=['1', '4', '8', '16', '32', '64'], width=60, state='readonly')
        self.num_epochs.set('32')
        self.num_epochs.grid(row=1, column=1, sticky='w')

        ctk.CTkLabel(self, text='Batch size: ', font=ctk.CTkFont(size=20, weight="bold")).grid(row=1, column=2, sticky='e')
        self.batch_size = ctk.CTkComboBox(self, values=['1', '4', '8', '16', '32', '64', '128', '256'], width=60, state='readonly')
        self.batch_size.set('64')
        self.batch_size.grid(row=1, column=3, sticky='w')
