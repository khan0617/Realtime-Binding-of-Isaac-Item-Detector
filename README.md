# Binding of Isaac Item Classifier 
Ever played the Binding of Isaac and didn't know what a new item does? Feel like installing the item descriptions mod is too easy of a solution? This project is for you!

[IN PROGRESS] Goals of this project:
- Learn webscraping by scraping wikis for item info such as ID, image, description (even though Isaac datasets exist already).
- Create a robust training set of images by augmenting the images (rotations, mirroring them, skewing etc) and putting them against a variety of backgrounds (such as Basement, Cellar, Womb etc.)
- Select a realtime object classifier model, such as YOLO, and implement this from scratch in PyTorch.
- Train the model using the augmented dataset.
- Figure out how to screen record `isaac.exe` from python, so that I can feed maybe one frame per second to the object detection model.
- Create a web-app with a Flask backend and React frontend (never used React so I want to learn) to display the feed, the bounding box detection of the model and the predicted item, with its description and info.