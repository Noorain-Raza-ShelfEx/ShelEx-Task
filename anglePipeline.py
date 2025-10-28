import numpy as np
import cv2
from ultralytics import YOLO
import os
import math






class SectionAnglePipeline:
    """ This pipeline predicts the sections masks and find out the angle
        between top left and top right points of top shelf of the cooler.
    Args:
        image_dir: Path to image file or directory to pass to YOLO.
        weight_path: Path to the YOLO weight file (e.g. last.pt).
    """

    def __init__(self, image_dir: str, weight_path: str, output_dir: str):
        self.image_dir = image_dir
        self.weight_path = weight_path
        self.output_dir = output_dir

    def predict_sections(self):
        """ Run the YOLO model on image_dir and return the results.

        Loads the model.
        Returns:
            A list of results
        """
        model = YOLO(self.weight_path)
        results = model(self.image_dir, conf=0.5, save=False, classes=[2])
        return results
    


    def get_left_right_corner_points(self, result, img_name):
        """ This function takes result of prediction and return the 
            top left and right corner point of top shelf of the cooler.

        Returns:
            left_min_point : list[x,y]
            right_min_point : list[x,y]
        """

        if result.masks is None:
            print(f"❌ No shelf masks detected in {img_name}")
            return None

        masks = result.masks.xy

        # -----------------------------
        # 1. Sort shelves top to bottom
        # -----------------------------
        shelf_data = []
        for m in masks:
            m_int = np.array(m, dtype=np.int32)
            y_mean = np.mean(m_int[:, 1])
            shelf_data.append((y_mean, m_int))
        shelf_data.sort(key=lambda x: x[0])

        # -----------------------------
        # 2. Take top shelf and calculate convex hull
        # -----------------------------
        top_shelf_mask = shelf_data[0][1]
        hull = cv2.convexHull(top_shelf_mask)
        hull_points = hull.reshape(-1, 2)

        # -----------------------------
        # 3. Calculate midpoint and shelf width
        # -----------------------------
        x_coords = hull_points[:, 0]
        y_coords = hull_points[:, 1]

        mid_x = np.mean(x_coords)
        mid_y = np.mean(y_coords)
        # min_y = np.min(y_coords)
        # Y = int((mid_y - min_y) / 2)
        shelf_length = np.max(x_coords) - np.min(x_coords)

        # -----------------------------
        # 4. Filter points
        # -----------------------------
        # Points with y less than mid_y (top half)
        top_points = [pt for pt in hull_points if pt[1] < mid_y]

        # Split into left and right halves (±20% of shelf length around center)
        left_threshold = mid_x - 0.2 * shelf_length
        right_threshold = mid_x + 0.2 * shelf_length

        left_points = [pt for pt in top_points if pt[0] < left_threshold]
        right_points = [pt for pt in top_points if pt[0] > right_threshold]

        # -----------------------------
        # 5. Find extreme points
        # -----------------------------
        left_min_point = None
        right_min_point = None

        if len(left_points) > 0:
            left_min_point = min(left_points, key=lambda p: p[1])  # min Y among left
        if len(right_points) > 0:
            right_min_point = min(right_points, key=lambda p: p[1])  # min Y among right

        
        return left_min_point, right_min_point




    def get_angle(self, point1, point2):
        """
        Calculate slope and angle (in degrees) between two points in image coordinates.
        
        Parameters:
            point1 (list): [x1, y1]
            point2 (list): [x2, y2]
            
        Returns:
            angle_deg (float): angle (in degrees) w.r.t the horizontal axis (image coordinate system)
        """
        x1, y1 = point1
        x2, y2 = point2

        dx = x2 - x1
        dy = y2 - y1

        # Handle vertical line case
        if dx == 0:
            slope = float('inf')
            # In image coordinates, downward means +ve angle (90°)
            angle_deg = 90.0 if dy > 0 else -90.0
        else:
            slope = dy / dx
            # invert dy to account for image y-axis direction
            angle_rad = math.atan2(-dy, dx)
            angle_deg = math.degrees(angle_rad)
        
        return angle_deg
    

    def draw_angle_ref_line(self, image_path, angle, left_pt, right_pt, output_dir):

        os.makedirs(output_dir, exist_ok=True)


    def run(self):
        """Execute prediction and get_left_right_corner_points functions for each result."""
        results = self.predict_sections()

        for res in results:
            image_path = res.path
            image_name = image_path.split(os.path.sep)[-1]

            img = cv2.imread(image_path)

            left_corner_pt, right_corner_pt = self.get_left_right_corner_points(res, image_name)

            # If all four keypoints are present (cooler is fully visible), find the angle between two top corner points.
            if left_corner_pt is not None and right_corner_pt is not None :
                print(f"finding angle for the image : {image_name}")
                print(f"left pt : {left_corner_pt} right pt : {right_corner_pt}")

                # angle calcualtion function
                angle = self.get_angle(left_corner_pt, right_corner_pt)

                if abs(angle) > 5.0 :
                    print(f"Retake the image as angle is greater than 5 degree. angle : {angle}")
                else:
                    print(f"Perfect shot.... angle : {angle}")


                # visualise and save the image
                self.draw_angle_ref_line(image_path, angle, left_corner_pt, right_corner_pt, output_dir)

            else:
                print(f"retake the image : {image_name}")

    


if __name__ == '__main__':
    image_dir = r"keyPoints_detection\vbl_images"
    model_weight = r"vbl\vbl weights\segmentation\segment_last.pt"
    output_dir = r"angle_output\output-1"

    pipeline = SectionAnglePipeline(image_dir=image_dir, weight_path=model_weight, output_dir = output_dir)

    print("pipeline start....")
    pipeline.run()
    print("pipeline finished.")


