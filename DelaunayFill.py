#!/usr/bin/env python3

import os # for os.path.plittext (remove .png from filename)
import PIL
import numpy as np
import scipy.ndimage
from scipy.spatial import Delaunay
from matplotlib import pyplot
import math
import argparse

PIL.Image.MAX_IMAGE_PIXELS = None # disable decompression bomb error

# empty config: (False, (r,g,b))
#               (True,  alpha)

# global vars, get set by command line parser
file_name = "Ansicht Ost_0.25.tif"
empty_config = (True, 0) #empty_config = (False, (255, 0, 255))

# still not changeable by command line argumets
max_distance = 10 # in pixels
chunk_size = 1000 # chunk_size² (output) pixels per chunk

def compute_is_empty(img):
	(use_alpha, value) = empty_config
	if use_alpha:
		print(f"Compute empty pixels using alpha = {value} ...")
		is_empty = img[:,:,3] == value
	else:
		print(f"Compute empty pixels using rgb = {value} ...")
		(r,g,b) = value
		is_empty = (img[:,:,0] == r) & (img[:,:,1] == g) & (img[:,:,2] == b)
	print(f"is_empty uses {humansize(is_empty.nbytes)}.")
	return is_empty


def compute_is_vertex(img, is_empty):
	print(f"Compute is_vertex pixels without optimization")

	is_vertex = ~is_empty
	# TODO: optimize away points inside filled surface

	print(f"is_vertex uses {humansize(is_vertex.nbytes)}.")
	return is_vertex

def encode_image(img):
	""" return (vertices, is_empty, is_vertex, colors) as numpy arrays """

	print("in encode_image:")

	#(h,w) = img.shape[:2]
	(use_alpha, value) = empty_config

	# compute is_empty numpy array
	is_empty = compute_is_empty(img)

	# compute is_vertex numpy array
	is_vertex = compute_is_vertex(img, is_empty)

	print("compute number of vertices...")
	vertex_count = np.sum(is_vertex)
	if vertex_count == 0:
		print("Warning: no vertices in this chunk, leave empty.")
		return (None, None, None, None)
		
	print(f"image contains {vertex_count} vertices.")

	print(f"compute vertex coordinates...")
	# [y,x] arrays
	vertices = np.argwhere(is_vertex).astype(np.uint16)
	print(f"vertices uses {humansize(vertices.nbytes)}.")

	print(f"extract vertex colors...")
	
	colors = np.zeros((vertex_count, 3), dtype=np.uint8)
	colors[:,0] = np.extract(is_vertex, img[:,:,0])
	colors[:,1] = np.extract(is_vertex, img[:,:,1])
	colors[:,2] = np.extract(is_vertex, img[:,:,2])

	print(f"colors uses {humansize(colors.nbytes)}.")

	return (vertices, is_empty, is_vertex, colors)

def fill_image(src_img, dst_img, colors, triangulation, is_empty, is_vertex, dst_offset):
	(dst_h, dst_w) = dst_img.shape[:2]
	should_interpolate = is_empty[
		dst_offset[0] : dst_h + dst_offset[0],
		dst_offset[1] : dst_w + dst_offset[1]]

	print("Search for triangles to interpolate from...")
	interpolation_coords = np.argwhere(should_interpolate)
	simplex_ids = triangulation.find_simplex(interpolation_coords + dst_offset)

	print("compute output colors")
	for i in range(len(interpolation_coords)):
		(y,x) = interpolation_coords[i]
		simplex_id = simplex_ids[i]

		# skip points outside of triangulation
		if simplex_id == -1:
			should_interpolate[y,x] = False
			continue

		# read triangle values
		simplex_vertexids = triangulation.simplices[simplex_id]
		simplex_vertices = triangulation.points[simplex_vertexids]
		
		(p1, p2, p3) = simplex_vertices

		v1 = p2 - p1
		v2 = p3 - p2
		v3 = p1 - p3

		# skip if triangle is too large
		if max(	np.linalg.norm(v1),
				np.linalg.norm(v1),
				np.linalg.norm(v3) ) > max_distance:
			should_interpolate[y,x] = False
			continue

		# compute barycentric coordinates using affine transformation
		# provided by scipy Delaunay
		barycentric = np.dot(
			triangulation.transform[simplex_id,:2],
			np.array([y,x]) + dst_offset - triangulation.transform[simplex_id,2])


		# interpolate final color
		v1i = (int(p1[0]), int(p1[1]))
		v2i = (int(p2[0]), int(p2[1]))
		v3i = (int(p3[0]), int(p3[1]))

		dst_img[y,x] =	barycentric[0] * src_img[v1i] \
			+			barycentric[1] * src_img[v2i] \
			+ (1 - barycentric[0] - barycentric[1]) * src_img[v3i]

def parse_args():
	parser = argparse.ArgumentParser(description='Fill holes in Images using delaunay triangulation. This program splits work into smaller chunks to be able to parallelize the workload and overcome memory limitations.')
	parser.add_argument('-a','--use-alpha', help='use alpha color to specify empty regions')
	parser.add_argument('-r','--use-rgb', nargs=3, help='use alpha color to specify empty regions')
	parser.add_argument('-c','--chunk-size', help='set size of blocks in which the image is computed. smaller chunks reduce memory usage and triangulation work, but increases overhead due to smaller arrays')
	parser.add_argument('-d','--max-distance', help='maximal distance between two points for interpolation')
	parser.add_argument('filename', help="the image to fill the holes")
	args = parser.parse_args()

	if args.use_alpha is not None:
		empty_config = (True, int(args.use_alpha))
	elif args.use_rgb is not None:
		empty_config = (False, (int(args.use_rgb[0]), int(args.use_rgb[1]), int(args.use_rgb[2])))
	else:
		if input("Use alpha channel to decide if points are empty? (y/n)").lower() == 'y':
			empty_config = (True, int(input("Enter alpha value for empty cells (probably 0)")))
		else:
			empty_config = (False, (
				int(input("Enter read value")),
				int(input("Enter green value")),
				int(input("Enter blue value"))
			))
	file_name = args.filename

	if args.chunk_size is not None:
		global chunk_size
		chunk_size = int(args.chunk_size)
		print(f"use chunk size {chunk_size}")
	if args.max_distance is not None:
		global max_distance
		max_distance = int(args.max_distance)
		print(f"use max distance {max_distance}")

def process_chunk(in_img, out_img, dst_offset):
	print(f"dst_offset: {dst_offset}, in_img.shape: {in_img.shape}, out_img.shape: {out_img.shape}")

	(vertices, is_empty, is_vertex, colors) = encode_image(in_img)
	if vertices is None:
		return

	print("Compute delaunay triangulation...")
	triangulation = Delaunay(vertices)
	fill_image(in_img, out_img, colors, triangulation, is_empty, is_vertex, dst_offset)


def main():
	parse_args()

	print(f"load image {file_name}...")
	img = np.array(PIL.Image.open(file_name))
	(h,w,channels) = img.shape
	print(f"Image size: w={w}, h={h}, channels={channels}, dtype={img.dtype}, memory={humansize(img.nbytes)}")

	dst_img = img.copy()
	chunk_count_x = math.ceil(w / chunk_size)
	chunk_count_y = math.ceil(h / chunk_size)

	for chunk_y in range(chunk_count_y):
		for chunk_x in range(chunk_count_x):
			print(f"= {int(100.0*(chunk_y * chunk_count_x + chunk_x) / (chunk_count_x * chunk_count_y))}% ===============================")
			print(f"=== fill chunk x:{chunk_x+1}/{chunk_count_x} y:{chunk_y+1}/{chunk_count_y} ===")
			
			src_img_chunk =     img[max(chunk_y * chunk_size - max_distance, 0) : (chunk_y + 1) * chunk_size + max_distance,
									max(chunk_x * chunk_size - max_distance, 0) : (chunk_x + 1) * chunk_size + max_distance]
			dst_img_chunk = dst_img[chunk_y * chunk_size                        : (chunk_y + 1) * chunk_size,
									chunk_x * chunk_size                        : (chunk_x + 1) * chunk_size]
			dst_offset = (chunk_y * chunk_size - max(chunk_y * chunk_size - max_distance, 0),
			              chunk_x * chunk_size - max(chunk_x * chunk_size - max_distance, 0))

			process_chunk(src_img_chunk, dst_img_chunk, dst_offset)

		#print("Show graph..")
		#pyplot.imshow(dst_img)
		#pyplot.show()
		#print("done.")

	dst_img_pil = PIL.Image.fromarray(dst_img)
	dst_img_pil.save(os.path.splitext(file_name)[0] + "_interpolated.png", "PNG")

	pyplot.imshow(dst_img)
	pyplot.show()

# ------- other stuff

suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
def humansize(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.2f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

if __name__ == "__main__":
	main()
