#https://www.paddleocr.ai/v3.3.2/en/version3.x/pipeline_usage/PP-StructureV3.html#22-python-script-integration

from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayout_plus-L")
output = model.predict("layout.jpg", batch_size=1, layout_nms=True)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")

