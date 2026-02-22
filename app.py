from paddleocr import PPStructureV3

pipeline = PPStructureV3(enable_mkldnn=False)

output = pipeline.predict("./sample_id.png")
for res in output:
    res.print() ## Print the structured prediction output
    res.save_to_json(save_path="output") ## Save the current image's structured result in JSON format
    res.save_to_markdown(save_path="output") ## Save the current image's result in Markdown format