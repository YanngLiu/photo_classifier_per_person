# Photo classifier per person

## What is it
This is a classifier which can generate symbolic link of certain person's photo files in specified folder, to make the selection of certain person's photo easier. 
It's based on Adam Geitgey's Face Recognition project.
The purpose of this project is to make selection of certian person's photo easier, giving you have lots of photos and mutliple directories hirarchies.
Maybe you want to choose some photoes to print out, but some product, such as Google Photo, needs to upload photo, and sometimes it just not work very quickly, or even cannot work in some location.

## Features
Classify photos according to the persons who we pay attention, the result is each concerned person has its own folder, its solo photo will have symbol link there, also a mutliple person photo foler, where also contains symbol link to the real photo.

## Installstion

### Requirements
  * Same as [Face Recognition](https://github.com/ageitgey/face_recognition)

## Usage
Download the code of this project, run it with arguments:
```bash
python3 _path_to_classifier.py_ --cpus -1 _path_to_symbollink_folder_ _path_to_single_target_persons_photoes_folder_ _path_to_photoes_to_classify_
```
When the classification is done, you can select certain photoes' symbol link files to copy to some folderA.
After that you can run `cp -rL folderA folderB` to copy these selected symbol link files' photoes to some folderB, then you can use them to publish your album or do whatever you want.
