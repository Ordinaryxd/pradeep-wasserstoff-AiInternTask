from utils.preprocessing import preprocess_image
from models.segmentation_model import segment_objects
from models.identification_model import identify_objects
from utils.postprocessing import analyze_objects
from utils.data_mapping import generate_summary_table
from utils.visualization import save_visualizations

def main(image_path):
    # Step 1: Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Step 2: Segment objects in the image
    segmentation_masks = segment_objects(preprocessed_image)

    # Step 3: Identify the segmented objects
    identified_objects = identify_objects(segmentation_masks, preprocessed_image)

    # Step 4: Analyze the identified objects
    analysis_results = analyze_objects(identified_objects, preprocessed_image)

    # Step 5: Generate the summary table
    summary_table = generate_summary_table(analysis_results)
    
    # Step 6: Save visualizations and results
    save_visualizations(image_path, segmentation_masks, identified_objects, summary_table)

if __name__ == "__main__":
    image_path = "data/input_images/sample_image.jpg"
    main(image_path)
