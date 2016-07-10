from urllib.request import urlopen, Request
from http.client import HTTPException
from urllib.error import URLError
import os
import shutil
import json
from collections import OrderedDict

def anchor(heading):
    return heading.replace(" ", "")


root_dir = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
            )
        )
json_dir = root_dir + "/json"
output_dir = root_dir + '/output'
with open(json_dir + '/links.json') as data_file:
    json_dict = json.load(data_file)

links = json_dict['links']

with open(json_dir + '/structure.json') as data_file:
    json_dict = json.load(data_file, object_pairs_hook=OrderedDict)

structure = json_dict['structure']

# Collect headings and tags for comparison at end of script
link_headings = set([])
for link in links:
    for tag in link['tags']:
        link_headings.add(tag)

# replace heading names with dictionaries to hold links
# and create inverse dictionary of headings -> folders
structure_headings = set([])
headings_to_folders = {}
for folder, values in structure.items():
    a = OrderedDict()
    for heading in values['headings']:
        headings_to_folders[heading] = folder
        structure_headings.add(heading)
        a[heading] = []
    values['headings'] = a
    

# delete old output
dirs = os.walk(output_dir)
for directory in next(dirs)[1]:
    shutil.rmtree(output_dir + "/" + directory)

try:
    os.remove(output_dir + "/README.md")
except FileNotFoundError:
    print("Could not delete the root README.md. The file was not found. " +
            "Don't worry about it.")


# for each link, check the response, and put it under the appropriate heading
for link in links:
    url_err = False
    try:
        req = Request(link['url'], headers={'User-Agent' : "Magic Browser"})
        res = urlopen(req)
        if res.status != 200:
            url_err = True
    except HTTPException:
        url_err = True
        print("HTTPException for " + link['url'])
    link['url_err'] = url_err
    for tag in link['tags']:
        structure[headings_to_folders[tag]]['headings'][tag].append(link)


"""
for link in links:
    url_err = False
    try:
        response = urlopen(link['url'])
    except URLError:
        url_err = True
    link['url_err'] = url_err
    for tag in link['tags']:
        structure[headings_to_folders[tag]]['headings'][tag].append(link)
"""

# Generate README.md's
# for each item in structure, create folder, README.md, and populate
with open(output_dir + "/README.md","a+") as toc:
    toc.write("#Free Online Data Science Resources  \n")
    toc.write("##Table of Contents  \n")
    for folder, item in structure.items():
        toc.write("###[" + item['title'] + "](./" + folder + ")  \n")
        new_folder = output_dir + "/" + folder
        os.mkdir(new_folder)
        with open(new_folder + "/README.md","a+") as f:
            f.write("#" + item['title'] + '  \n')
            f.write("## Local Table of Contents  \n")
            f.write("[(Back to Master Table of Contents)](../)  \n")
            for heading in item['headings'].keys():
                f.write("[" + heading + "](#" + anchor(heading) + ")  \n")
            for heading, hlinks in item['headings'].items():
                toc.write("[" + heading + "](" + folder + "#" + anchor(heading)
                        + ")  \n")
                f.write("## <a name=\"" + anchor(heading) + "\"></a>" + 
                        heading + "  \n\n")
                for link in hlinks:
                    f.write("[" + link['title'] + "](" + link['url'] + ")")
                    if link['url_err']:
                        f.write(" (URL Failure)")
                    f.write("  \n")
                    if 'author' in link:
                        f.write("by " + link['author'] + "  \n")
                    f.write(link['description'] + "  \n")
                    if len(link['tags']) > 1:
                        f.write("Other tags: " )
                        tags = link['tags'][:]
                        tags.remove(heading)
                        for tag in tags[:-1]:
                           f.write("[" + tag + "](../" + headings_to_folders[tag] + "#" + anchor(tag) + "), ")
                        f.write("[" + tags[-1] + "](../" + headings_to_folders[tags[-1]] + "#" + anchor(tags[-1]) + ") ")
                        f.write("  \n")
                    f.write("  \n")

print("Unused tags: ", str(link_headings.difference(structure_headings))) 
print("Unused headings: ", str(structure_headings.difference(link_headings))) 

# Check for duplicate links
# Add special logic for URL checking of edx
# print list of url errors
# check for each link not being listed on more than one page
