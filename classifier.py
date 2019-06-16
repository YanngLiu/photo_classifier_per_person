from __future__ import print_function
import face_recognition
import os
from pathlib import Path
import numpy as np
import collections
import pathlib
import sys
import itertools
import click
import multiprocessing

MULTI_FOLDER_NAME='multi'

def init(link_file_folder,known_people_folder):
    personEncodings,personNames=[],[]
    for s in os.listdir(known_people_folder):
        dotIdx=s.index('.')
        name=s[:dotIdx]
        personNames.append(name)
        linkUrl=link_file_folder+name+'/'+MULTI_FOLDER_NAME+'/'
        if not os.path.isdir(linkUrl) or not os.path.exists(linkUrl):
            pathlib.Path(linkUrl).mkdir(parents=True, exist_ok=True)
    N=len(personNames)
    sortedPersons=sorted(personNames)
    allPersons=''.join(sortedPersons)
    for s in os.listdir(known_people_folder):
        img=face_recognition.load_image_file(known_people_folder+s)
        imgEncoding = face_recognition.face_encodings(img)[0]
        personEncodings.append(imgEncoding)
    sortedPersonEncodings=[personEncodings[personNames.index(name)] for name in sortedPersons]
    personEncodings=sortedPersonEncodings
    personNames=sortedPersons
    print('persons: {} {}'.format(N, personNames))
    return personEncodings,personNames

def printProgress(show_progress,txt):
    if not show_progress: return
    print(txt)
def classify_image(photoPath,link_file_folder,personEncodings,personNames,tolerance=0.45,show_progress=False):
    if photoPath.is_symlink():
        printProgress(show_progress,'link')
        return
    photoFileName=photoPath.name
    
    try:
        img=face_recognition.load_image_file(photoPath)
        imgEncodings = face_recognition.face_encodings(img)
        # if no person, ignore
        if len(imgEncodings)==0:
            printProgress(show_progress,'no person')
            return
        multiplePerson=len(imgEncodings)>1
        matched=False
        # for each person in the photo
        for imgEncoding in imgEncodings:
            distances = face_recognition.face_distance(personEncodings, imgEncoding)
            minDistanceIdx=np.argmin(distances)
            if distances[minDistanceIdx] > tolerance:
                continue
            matched=True
            personName=personNames[minDistanceIdx]
            baseFolder=link_file_folder+personName+'/'
            multiFolder=baseFolder+MULTI_FOLDER_NAME+'/'
            if multiplePerson:
                dst=multiFolder+personName+photoFileName
                printProgress(show_progress,'{} multiple photo'.format(personName))
            else:
                dst=baseFolder+personName+photoFileName
                printProgress(show_progress,'{} single photo'.format(personName))
            if not os.path.exists(dst):
                os.symlink(photoPath, dst)
        if not matched: printProgress(show_progress,'Not matched')
    except Exception as e:
        print(photoPath, e)

def classify(link_file_folder,to_check_folder,personEncodings,personNames,number_of_cpus,tolerance,show_progress):
    photoPaths=list(Path(to_check_folder).rglob("*.[jJ][pP][gG]"))
    print( '{} photos to classify.'.format(len(photoPaths)))
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you don't use 'forkserver'
    context = multiprocessing
    if "forkserver" in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context("forkserver")

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        photoPaths,
        itertools.repeat(link_file_folder),
        itertools.repeat(personEncodings),
        itertools.repeat(personNames),
        itertools.repeat(tolerance),
        itertools.repeat(show_progress)
    )

    pool.starmap(classify_image, function_parameters)        


@click.command()
@click.argument('link_file_folder') #/home/yang/acp/
@click.argument('known_people_folder') #/media/3TNew/pics/samples/
@click.argument('to_check_folder') #/media/3TNew/pics/
@click.option('--cpus', default=1, help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"')
@click.option('--tolerance', default=0.45, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
@click.option('--show_progress', default=False, type=bool, help='Output progress.')
def main(link_file_folder, known_people_folder, to_check_folder, cpus, tolerance, show_progress):
    # Multi-core processing only supported on Python 3.4 or greater
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo("WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!")
        cpus = 1
    personEncodings,personNames=init(link_file_folder,known_people_folder)
    classify(link_file_folder,to_check_folder,personEncodings,personNames,cpus,tolerance,show_progress)
if __name__=="__main__":
    main()
