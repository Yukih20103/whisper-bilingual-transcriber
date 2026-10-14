from main_transcribe import load_config_and_process_transcribe 
import flet as ft
from flet import View, Page, AppBar, Text, Column, Row, Icon, Icons
from flet import TextField, Dropdown, FilePicker
from flet import Button, SegmentedButton
import asyncio
import sys
import queue

def main(page: ft.Page):
    page.title = "Whisper Transcribe 音訊轉文字"
    page.window.width = 1000
    page.window.height = 700
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.window.center()
    file_picker_for_dir = FilePicker()

    TopBar = AppBar(title = Text("Whisper Transcriber (Chinese and English)", size=20),
                    bgcolor="blue",
                    leading_width=10,
                    )

    Ori_whisper = Dropdown(
        label =Text("Original Whisper Model", size=20),
        value="large-v3",
        options=[
            ft.dropdown.Option("tiny"),
            ft.dropdown.Option("base"),
            ft.dropdown.Option("small"),
            ft.dropdown.Option("medium"),
            ft.dropdown.Option("large"),
            ft.dropdown.Option("large-v3"),
        ],
        width = 200,
    )

    class TrueFalseButton(Column):
        def __init__(self, text):
            super().__init__()  # Column's init
            self.text = text
            self.segmented = SegmentedButton(
                    selected_icon=Icon(Icons.CHECK_SHARP),
                    selected=["False"],
                    allow_empty_selection=False,
                    allow_multiple_selection=False,
                    segments=[
                        ft.Segment(value="True", label=ft.Text("True")),
                        ft.Segment(value="False", label=ft.Text("False")),
                    ],
                    width=200,
                )
            self.controls = [
                ft.Text(self.text, size=15),
                self.segmented,
            ]

    def save_term_path(path):
        selected_term_file["path"] = path

    def save_source_path(path):
        selected_source_file["path"] = path

    class FilePickerButton(ft.Column):
        def __init__(self, label, extensions, on_file_selected):
            super().__init__(spacing=5)
            self.extensions = extensions
            self.on_file_selected = on_file_selected
            self.file_picker = ft.FilePicker()
            self.status_text = ft.Text("No file selected", color="gray", size=14)

            self.button = ft.Button(
                content=label,
                icon=ft.Icons.UPLOAD_FILE,
                on_click=self._handle_click,
            )

            self.controls = [self.button, self.status_text]

        async def _handle_click(self, e):
            if self.file_picker not in self.page.services:
                self.page.services.append(self.file_picker)
                self.page.update()

            files = await self.file_picker.pick_files(
                allow_multiple=False,
                file_type=ft.FilePickerFileType.CUSTOM,
                allowed_extensions=self.extensions,
            )
            if files:
                self.status_text.value = f"Selected: {files[0].name}"
                self.status_text.color = "green"
                self.on_file_selected(files[0].path)
            else:
                self.status_text.value = "No file selected"
                self.status_text.color = "gray"
            self.page.update()

    async def pick_directory(e):
        if file_picker_for_dir not in page.services:
            page.services.append(file_picker_for_dir)
            page.update()

        path = await file_picker_for_dir.get_directory_path(
            dialog_title="Select Output Path"
        )
        if path:
            selected_output_dir["path"] = path
            dir_status_text.value = f"Output path selected: {path}"
            page.update()

    dir_status_text = Text("No output path selected", color="gray", size=14)
    result_text = Text("", size=14)

    selected_source_file = {"path": None}
    selected_term_file = {"path": None}
    selected_output_dir = {"path": None}

    log_view = ft.ListView(
        expand=True,
        spacing=2,
        auto_scroll=True,
        padding=10,
    )
    log_container = ft.Container(
        content=log_view,
        bgcolor="#1e1e1e",
        border_radius=8,
        padding=5,
        height=230,
        width=800,
    )

    log_queue = queue.Queue()
    class QueueStream:
        def __init__(self, q):
            self.q = q
        def write(self, text):
            if text.strip():
                self.q.put(text.rstrip())
        def flush(self):
            pass

    sys.stdout = QueueStream(log_queue)
    def drain_log_queue():
        updated = False
        while not log_queue.empty():
            text = log_queue.get()
            log_view.controls.append(
                ft.Text(text, color="#d4d4d4", size=15, selectable=True)
            )
            updated = True
        if updated:
            page.update()

    async def poll_log_queue():
        while True:
            drain_log_queue()
            await asyncio.sleep(0.2)

    async def run_backend_process(e):
        source_path = selected_source_file["path"]
        term_path = selected_term_file["path"]
        output_dir = selected_output_dir["path"]

        model = Ori_whisper.value
        one_speaker = one_speaker_button.segmented.selected[0] == "True"
        #use_finetuned = use_finetuned_button.segmented.selected[0] == "True"
        traditional_chinese = traditional_chinese_button.segmented.selected[0] == "True"
        use_finetuned = False
        
        if not term_path:
            term_path = ""

        if not output_dir or not source_path:
            result_text.value = "❌ Please fill in all fields (except terms.txt)"
            result_text.color = "red"
            page.update()
            return

        start_button.disabled = True
        start_button.text = "Processing..."
        page.update()
    
        try:
            final_path = await asyncio.to_thread(
            load_config_and_process_transcribe,
            source_path,
            output_dir,
            term_path,
            model,
            use_finetuned,
            one_speaker,
            traditional_chinese,
        )
            result_text.value = f"✅ Done! Output saved to {final_path}."
            result_text.color = "green"
        except Exception as ex:
            result_text.value = f"❌ Error: {ex}"
            result_text.color = "red"
        finally:
            start_button.disabled = False
            start_button.text = "Start Transcription"
            page.update()

    source_picker = FilePickerButton(
        label="Select an audio or video file (wav, mp4)",
        extensions=["wav", "mp4"],
        on_file_selected=save_source_path,
    )

    term_picker = FilePickerButton(
        label="(Optional) Select term.txt for Chinese term matching",
        extensions=["txt"],
        on_file_selected=save_term_path,
    )

    select_dir_button = Button(
        "Select Output Directory",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=pick_directory,
        )

    use_finetuned_button = TrueFalseButton("Use finetuned model")   
    one_speaker_button =  TrueFalseButton("The audio only has one speaker/language")   
    traditional_chinese_button =  TrueFalseButton("Output traditional Chinese character (if not then simplified Chinese)")  

    start_button = Button(
        "Start Transcription",
        icon=ft.Icons.PLAY_ARROW,
        on_click=run_backend_process,
)

    parameters = Column([
        Ori_whisper,
        #use_finetuned_button,
        one_speaker_button,
        traditional_chinese_button,
    ],
    spacing=10,)

    paths = Column([
        source_picker,
        term_picker,
        select_dir_button,
        dir_status_text, 
        start_button,
        result_text,
    ],
    spacing=10,)
    
    page.add(
        TopBar,
        Row([
            parameters,
            paths,
        ],
        spacing=100,
        alignment=ft.MainAxisAlignment.CENTER,),
        log_container,
    )

    page.run_task(poll_log_queue)

ft.run(main)