from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
from core.operation_factory import build_operation
from core.operations import MultiOutputOperation
class ImageWorker(QThread):
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self, image: np.ndarray, recipe: list, parent=None):
        super().__init__(parent)
        self.image = image
        self.recipe = recipe # This is now a list[OperationConfig]

    def run(self):
        try:
            if not self.recipe:
                self.result_ready.emit({"action": "None", "data": self.image.copy()})
                return

            current_image = self.image.copy()
            final_multi_buffer = None

            for config in self.recipe:
                # 1. Ask the Factory for the OpenCV execution class
                operation_instance = build_operation(config)

                # 2. Apply the mathematical logic
                if isinstance(operation_instance, MultiOutputOperation):
                    result_dict = operation_instance.apply_extended(current_image)
                    final_multi_buffer = result_dict
                    current_image = result_dict["magnitude"]
                else:
                    current_image = operation_instance.apply(current_image)
                    final_multi_buffer = None

            # 3. Emit the final payload
            # (We can use the class name of the last config as the UI action hint)
            final_action_hint = type(operation_instance).__name__
            
            self.result_ready.emit({
                "action": final_action_hint,
                "data": final_multi_buffer if final_multi_buffer else current_image
            })

        except Exception as e:
            self.error_occurred.emit(f"Worker Error: {str(e)}")
