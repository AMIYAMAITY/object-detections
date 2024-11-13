from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import requests
import base64
import json
import cv2


app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    url = 'http://localhost:8080/add_json'
    _headers = {"Content-Type": "application/json; charset=utf-8"}


    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the image in memory
        image = Image.open(io.BytesIO(file.read()))
        
        # Optionally convert the image to a NumPy array
        image_array = np.array(image)
        jpg_img = cv2.imencode('.jpg', image_array)
        b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')

        params = {
            'image_str': b64_string
        }

        response = None
        try:
            response = requests.post(url, headers=_headers, json=params)
        except Exception as ex:
            print("Eception: ", ex)
            return jsonify({
                'message': 'Something went wrong'
            }), 500
        
            pass


        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        if response.status_code == 200 :
            data = response.json()  # Parse JSON response if applicable
            print("Response data:", data)
            print("shape: ", image_array.shape)
            
            image_file_path = None
            json_file_path = None
            if 'imageinfo' in data:
                image_file_path = data['imageinfo']['image_path']
                json_file_path = data['imageinfo']['json_path']

                if 'detections' in data:
                    for keys in data['detections'].keys():
                        class_name = keys
                        for obj in data['detections'][keys]:
                            xmin = obj['xmin']
                            ymin = obj['ymin']
                            width = obj['width']
                            height = obj['height']
                            conf = obj['conf']

                            image_array = cv2.rectangle(image_array, (xmin, ymin), (xmin + width, ymin + height), (255, 255, 0), 2)
                            text =  "{} {:.2f}".format(class_name, conf)
                            image_array = cv2.putText(image_array,text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

            image_file_path = image_file_path.replace(' ', '-')
            json_file_path = json_file_path.replace(' ', '-')
            if image_file_path is not None and json_file_path is not None:
                cv2.imwrite(image_file_path, image_array)
                with open(json_file_path, "w") as f:
                    json.dump(data, f, indent=4)

            return jsonify(data), 200
            
        else:
            return jsonify({
                'message': 'Something went wrong'
            }), 500
        
            print("Request failed with status code:", response.status_code)

        # Return some information about the image


       

    return jsonify({'error': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(debug=True)
