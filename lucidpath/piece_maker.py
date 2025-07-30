#!/gpfs0/arazi/projects/miniconda/bin/python3


###
###  Import libraries
###

# Import standard libraries
import os
import sys
import itertools

# Import third party libraries
import pandas as pd
import numpy as np

import scipy.cluster as cls
import scipy.spatial as spt

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.csgraph import connected_components

from invisible_cities.io.mcinfo_io import load_mchits_df as ldhits

###
###  Set variables
###

#Set variables
data_path = sys.argv[1]
file_name = sys.argv[2]


multitrack_distance = 30   #Distance to main track to be considered a satellite
branch_distance     = 5    #Distance to main track to be considered potentially a satellite
branch_extent_param = 20    #Extent of potential satellite to indicate it is actually a part of main track
cluster_size        = 10    #Used to determine how many key points to use for extended clusters
plane_cut_distance  = 2.5   #Distance between two slices indicating independent crossings of a plane



###
###  Define functions
###

def distance(zipped_coords):
  '''Return the distance between n-dimensional points
     using root sum squares.
  '''
  coord_squares = [(c1 - c2)**2 for c1, c2 in zipped_coords]
  return np.sqrt(sum(coord_squares))


def identify_keypoints(event_df):
  '''
  Reduce each crossing of the z-plane to a small number of key points
  The Z-axis is selected here because deconvolution, as of the time of this
    writing, does not operate on the z-axis. This results in unusual spacing
    in this dimension that makes for a natural break. Any axis could be chosen
    if there were no distinction in post deconvolution reconstruction.
  '''
  key_points = []

  for z_plane in event_df.Z.unique():
    #Consider each plane at a time
    plane_df = event_df[event_df.Z == z_plane]
    plane_df.reset_index(drop=True, inplace=True)

    #Sometimes there is only a single voxel on a given z-plane; ignore these
    if len(plane_df) > 1:
      #Determine how many times the track crosses this plane.
      pdist = spt.distance.pdist(plane_df[['X','Y']].values)
      Z = cls.hierarchy.linkage(pdist)
      cut_tree = cls.hierarchy.cut_tree(Z, height=plane_cut_distance)
      plane_df = plane_df.merge(pd.DataFrame(cut_tree, columns=['Cluster']), left_index=True, right_index=True)

      #Multiple crossings should have different points for the path
      for name, cluster_df in plane_df.groupby('Cluster'):
        #As above, we ignore very small clusters
        if len(cluster_df) > 2:
          #Look for the two most distant points from eachother in the cluster
          #Store distances between each set of points
          pdist = spt.distance.pdist(cluster_df[['X','Y']].values)
          d_mat = spt.distance.squareform(pdist)

          #Find indexes of the most distant points
          max_dist = max([max(each) for each in d_mat])
          first_idx = int([max(each) for each in d_mat].index(max_dist))
          second_idx = int(list(d_mat[first_idx]).index(max(d_mat[first_idx])))

          #Identify the relevant points
          p1 = (cluster_df.iloc[first_idx].X, cluster_df.iloc[first_idx].Y)
          p2 = (cluster_df.iloc[second_idx].X, cluster_df.iloc[second_idx].Y)

          #The points for the crossing is determined by the distance between
          #  these points and the parameter set above
          n_div = np.floor(distance(zip(p1,p2))/cluster_size)

          #If those points are more than twice cluster_size distance apart,
          #  we add multiple points for that single crossing.
          #The second condition prevents a divide by zero error.
          if (distance(zip(p1,p2)) > cluster_size) and abs(p1[0] - p2[0]) > 0:
            #A cluster is treated in 2D using the line between
            #  the most distant points to set additional points
            #The most distance points are calculated as line relative to x-axis
            m = (p1[1]-p2[1])/(p1[0]-p2[0])
            b = p1[1] - p1[0]*m

            min_x = min([p1[0],p2[0]])
            max_x = max([p1[0],p2[0]])
            x_range = max_x - min_x

            min_y = min([p1[1],p2[1]])
            max_y = max([p1[1],p2[1]])
            y_range = max_y - min_y

            #Each point along this line is added individually
            for section_num in range(int(n_div)):
              #points are set along the longer of the x or y axis.
              if x_range > y_range:
                low_cut = cluster_df.X > min_x + section_num*x_range/n_div
                high_cut = cluster_df.X < min_x + (section_num+1)*x_range/n_div
                section_df = cluster_df[low_cut & high_cut]
              elif y_range >= x_range:
                low_cut = cluster_df.Y > min_y + section_num*y_range/n_div
                high_cut = cluster_df.Y < min_y + (section_num+1)*y_range/n_div
                section_df = cluster_df[low_cut & high_cut]

              #Use the barycenter along each axis
              new_x = np.average(section_df.X, weights=section_df.E)
              new_y = np.average(section_df.Y, weights=section_df.E)
              new_z = np.average(cluster_df.Z)
              new_point = (new_x, new_y, new_z)
              key_points.append(new_point)

          #If the most distant points align with the x-axis
          elif p1[0] - p2[0] == 0:
            min_y = min([p1[1],p2[1]])
            max_y = max([p1[1],p2[1]])
            y_range = max_y - min_y

            for section_num in range(int(n_div)):
              low_cut = cluster_df.Y >= min_y + section_num*y_range/n_div
              high_cut = cluster_df.Y <= min_y + (section_num+1)*y_range/n_div
              section_df = cluster_df[low_cut & high_cut]
              new_x = np.average(section_df.X, weights=section_df.E)
              new_y = np.average(section_df.Y, weights=section_df.E)
              new_z = np.average(cluster_df.Z)
              new_point = (new_x, new_y, new_z)
              key_points.append(new_point)

          else:
            #If there is only a single key point for the crossing
            new_x = np.average(cluster_df.X, weights=cluster_df.E)
            new_y = np.average(cluster_df.Y, weights=cluster_df.E)
            new_z = np.average(cluster_df.Z)
            new_point = (new_x, new_y, new_z)
            key_points.append(new_point)

  return key_points


def end_df(pieces, short_version=True, connected=False):
  '''
  Provide information for each end point of each piece in 'pieces'.

  input:  list of lists containing n-dimensional points.
  output: pandas dataframe of end information


  The returned dataframe includes the following columns:

  Piece_Index  - Each piece in 'pieces' is given a sequential number
  End_Index    - The first point in a piece is 0, the final point is 1
  End_Point    - Coordinates of the point
  Near_Point   - Coordinates of the nearest point not on the same piece
  NP_Piece     - Piece_Index of the piece containing the nearest point
  NP_Index     - Index in the NP_Piece at which the Near_Point is located
  End_Distance - Distance between End_Point and Near_Point
  Piece_Length - Length along the piece


  If True, short_version returns only information for the nearest point
  to each end. If False, the dataframe contains information about the
  distance from each end to the nearest point of each other piece.

  If False, connected excludes 'near' points that are touching.
  '''
  #Initialize dataframe
  end_info_df = pd.DataFrame()

  #Compare each piece to all other pieces
  for piece_index, nearpiece_index in itertools.permutations(range(len(pieces)),int(2)):

    this_piece = pieces[piece_index]

    #End coordinates of the piece in question
    end_1 = pieces[piece_index][0]
    end_2 = pieces[piece_index][-1]

    #Calculate distance between each end and all points of one other piece
    #dist_list has two entries; the first is a list of distances to end_1
    #  and the second is a list of distances to end_2
    dist_list = spt.distance.cdist([end_1,end_2],pieces[nearpiece_index])
    #near_points is generated from the smallest distance from each of the
    # two lists and returning the distance and index
    near_points = [sorted(enumerate(each), key=lambda i: i[1])[0] for each in dist_list]

    #For each end, determine minimum distance from end to the other piece
    #Also record how far away the other end is at its nearest point
    for end_index, each_end in enumerate(near_points):
      #Returns 0th entry for first point and -1st for last point
      end_point = this_piece[-end_index]
      nearpoint_index = each_end[0]
      nearpoint = pieces[nearpiece_index][nearpoint_index]
      end_distance = each_end[1]
      piece_length = sum([distance(zip(this_piece[index-1],this_piece[index])) for index in range(len(this_piece))][1:])

      connection_df = pd.DataFrame(
  data=[[piece_index,end_index,end_point,nearpoint,nearpiece_index,nearpoint_index,end_distance,piece_length]],
  columns=['Piece_Index','End_Index','End_Point','Near_Point','NP_Piece','NP_Index','End_Distance','Piece_Length'])
      end_info_df = pd.concat([end_info_df, connection_df])

  #We only care about the closest points
  if len(end_info_df) == 0:
    return end_info_df
  if not connected:
    end_info_df = end_info_df[end_info_df.End_Distance > 0]
  if short_version:
    end_info_df = end_info_df.sort_values('End_Distance', ascending=True).groupby(['Piece_Index','End_Index']).head(int(1))
  elif not short_version:
    end_info_df = end_info_df.sort_values(['Piece_Index','End_Index','End_Distance'])

  #Sort and reset indexes
  end_info_df.sort_values(['Piece_Index','End_Index'], inplace=True)
  end_info_df.reset_index(drop=True, inplace=True)

  return end_info_df


def find_end_pairs(end_info_df):
  '''
  The Piece_Index and End_Index values are used to identify contiguous ends.

  input:  pandas dataframe generated by end_df() function with connected=True
  output: input dataframe with two additional columns identifying the connections among ends

  The returned dataset includes the following additional columns:

  This_End_Info - Combination of Piece_Index and End_Index into a tuple for comparison
  End_Pair_Info - False if end is not connected, otherwise gives piece and index for connection
  '''
  #Combine the Piece_Index and End_Index columns into a tuple
  this_end = end_info_df.apply(lambda i: (i.Piece_Index, i.End_Index), axis=int(1))
  this_end.name = 'This_End_Info'
  end_info_df = end_info_df.merge(this_end, left_index=True, right_index=True)

  #Do the same for the nearpoint if it is connected.
  end_pair = []

  for each_end in end_info_df.itertuples():
    if each_end.End_Distance == 0:
      if each_end.NP_Index == 0:
        end_pair.append((each_end.NP_Piece,0))
      elif each_end.NP_Index > 0:
        end_pair.append((each_end.NP_Piece, 1))
    else:
      end_pair.append(False)

  end_info_df = end_info_df.merge(pd.Series(end_pair, name='End_Pair_Info'), left_index=True, right_index=True)
  return end_info_df


def find_nodes(end_info_df):
  '''
  Contiguous end information is used to label nodes. Each end is associated with a
  node and the piece connects that node to the node associated with the other end.

  input:  pandas dataframe that has been processed with find_end_pairs() function
  output: input dataframe with two additional columns identifying node connections

  The returned dataframe includes the following additional columns:

  Node       - Node of end for this row
  Connected_Node - Node of the other end for the piece for this row
  '''
  #The index must be reset for the function to work as expected
  end_info_df.reset_index(drop=True, inplace=True)

  #The first node is always zero
  node_list = [0]

  for index, each_end in enumerate(end_info_df.itertuples()):
    #Always start with a new node
    if index == 0:
      continue

    #If the end for this row has already been assigned a node, use the previous assignment
    if (each_end.This_End_Info in list(end_info_df.This_End_Info.head(int(index)).values)):
      node_id = end_info_df[end_info_df.This_End_Info == each_end.This_End_Info].index.values[0]
      node_num = node_list[node_id]
      node_list.append(node_num)

    #If the end for this row is not connected to any other ends, it gets a new node designation
    elif (not each_end.End_Pair_Info):
      node_list.append(max(node_list) + 1)

    #If the end for this row has a connection, several conditions must be met before making a new node
    #If the end is the connection point of an end already examined, use that node
    elif (each_end.This_End_Info in list(end_info_df.End_Pair_Info.head(int(index)).values)):
      node_id = end_info_df[end_info_df.End_Pair_Info == each_end.This_End_Info].index.values[0]
      node_num = node_list[node_id]
      node_list.append(node_num)

    #If the end to which this end is connected has already been examined
    elif (each_end.End_Pair_Info in list(end_info_df.This_End_Info.head(int(index)).values)):
      node_id = end_info_df[end_info_df.This_End_Info == each_end.End_Pair_Info].index.values[0]
      node_num = node_list[node_id]
      node_list.append(node_num)

    #If the end to which this end is connected is already a part of an assigned
    elif (each_end.End_Pair_Info in list(end_info_df.End_Pair_Info.head(int(index)).values)):
      node_id = end_info_df[end_info_df.End_Pair_Info == each_end.End_Pair_Info].index.values[0]
      node_num = node_list[node_id]
      node_list.append(node_num)

    #If there are more than three pieces connected, check the end point
    elif (each_end.End_Point in list(end_info_df.End_Point.head(int(index)).values)):
      node_id = end_info_df[end_info_df.End_Point == each_end.End_Point].index.values[0]
      node_num = node_list[node_id]
      node_list.append(node_num)

    else:
      node_list.append(max(node_list) + 1)

  end_info_df = end_info_df.merge(pd.Series(node_list, name='Node'), left_index=True, right_index=True)

  #Determine which other nodes you can get to from each end
  #We can get this by just flipping the End_Index order for each piece
  connected_node_list = list(end_info_df.sort_values(['Piece_Index','End_Index'], ascending=[True, False]).Node.values)

  end_info_df = end_info_df.merge(pd.Series(connected_node_list, name='Connected_Node'), left_index=True, right_index=True)

  return end_info_df


def graphify(end_info_df):
  '''
  Produces a graph using the end_info_df nodes. The Piece_Length information is
  used to calculate the distance along all paths between each pair of nodes.

  input:  pandas dataframe that has been processed with find_nodes() function
  output: tuple of path data in matrix form

  Returned tuple has two matrix items. Column and row indexes correspond to node numbers:

  path distances - entries correspond to node-to-node path distance
  predecessors   - entries correspond to the previous node index along the shortest path
  '''
  #Python starts counting at 0
  number_of_nodes = end_info_df.Node.max() + 1

  #Use piece lengths to create distance graph
  graph_mat = np.zeros((number_of_nodes,number_of_nodes))
  for each_end in end_info_df.itertuples():
    graph_mat[each_end.Node][each_end.Connected_Node] = each_end.Piece_Length

  #Functions taken from scipy.sparse
  condensed_mat = csr_matrix(graph_mat)
  path_distances, predecessors = dijkstra(condensed_mat, return_predecessors = True)

  return path_distances, predecessors


def groupify(end_info_df):
  '''
  Contiguous end information is used to label groups. A group is a collection of
  pieces that are connected in the sense that each piece shares a point with at least
  one other piece in the group and no points with any piece not in the group.

  input:  pandas dataframe that has been processed with find_nodes() function
  output: input dataframe with two additional columns identifying connective groups

  The returned dataframe includes the following additional columns:

  Group  - Group of end for this row
  NP_Group - Group of the nearest point to this end
  '''
  #Python starts counting at 0
  number_of_nodes = end_info_df.Node.max() + 1

  #Use piece lengths to create distance graph
  graph_mat = np.zeros((number_of_nodes,number_of_nodes))
  for each_end in end_info_df.itertuples():
    graph_mat[each_end.Node][each_end.Connected_Node] = each_end.Piece_Length

  #Functions taken from scipy.sparse
  condensed_mat = csr_matrix(graph_mat)
  _, labels = connected_components(condensed_mat, directed=False)

  groups = list(zip(np.transpose(np.where(graph_mat)),labels))
  group_ids = []

  for each_end in end_info_df.itertuples():
    group_ids.append(labels[each_end.Node])

  end_info_df = end_info_df.merge(pd.Series(group_ids, name='Group'), left_index=True, right_index=True)

  NP_Group = []
  for each_end in end_info_df.itertuples():
    NP_Group.append(end_info_df[end_info_df.Piece_Index == each_end.NP_Piece].Group.values[0])

  end_info_df = end_info_df.merge(pd.Series(NP_Group, name='NP_Group'), left_index=True, right_index=True)

  return end_info_df


def join_pieces(pieces):
  '''
  Given a set of pieces, combine one set of contiguous pieces. If the input
  pieces are the same as the output, return zero

  input:  list of lists of x,y,z datapoints
  output: input list with two or three contiguous pieces joined
  '''
  #Clean pieces to make them junction to junction
  all_ends = end_df(pieces, short_version=True, connected=True)
  all_ends = find_end_pairs(all_ends)
  all_ends = find_nodes(all_ends)
  all_ends = groupify(all_ends)

  #Identify points that have three or more pieces ending on them
  end_nodes = all_ends[all_ends.End_Pair_Info == False].Node.values
  connect_count = all_ends.groupby('Node').count().sort_values(['Piece_Index'], ascending=False)
  junctions = connect_count[connect_count.Piece_Index >= 3].index.values

  stitched_pieces = []
  used_pieces = []

  #Identify pieces whose ends are neither junctions nor loose ends
  not_an_end = ~(all_ends.Node.isin(end_nodes))
  not_a_junction = ~(all_ends.Node.isin(junctions))
  middle_pieces = all_ends[not_an_end & not_a_junction]

  #If there are no middle pieces, move to next step
  if len(middle_pieces) == 0:
    return 0

  neighbor_info = middle_pieces.iloc[int(0)]

  piece_index = neighbor_info.Piece_Index
  neighbor_piece_index = neighbor_info.NP_Piece

  used_pieces.append(piece_index)

  #Using the end_index in this way places the relevant end as the last item in the piece
  this_piece = pieces[piece_index][::(-1)**np.abs(neighbor_info.End_Index - 1)]

  neighbor_exp = neighbor_info.End_Pair_Info[int(1)]
  neighbor_piece = pieces[neighbor_info.NP_Piece][::(-1)**(neighbor_exp)][1:]

  if not any([neighbor_piece[index] in this_piece for index in range(len(neighbor_piece))]):
    this_piece.extend(neighbor_piece)
    used_pieces.append(neighbor_piece_index)


  other_connection = middle_pieces[(middle_pieces.Piece_Index == piece_index) & ~ (middle_pieces.NP_Piece == neighbor_piece_index)]

  if len(other_connection) > 0:
    neighbor_info = other_connection.iloc[int(0)]

    neighbor_piece_index = neighbor_info.NP_Piece

    used_pieces.append(neighbor_piece_index)

    neighbor_exp = neighbor_info.End_Pair_Info[int(1)]
    neighbor_piece = pieces[neighbor_info.NP_Piece][::int((-1)**(neighbor_exp -1))][:int(-1)]

    neighbor_piece.extend(this_piece)

    this_piece = neighbor_piece

  stitched_pieces.append(this_piece)

  #The stitched pieces plus those that have not been addressed are considered for ...
  for index in range(len(pieces)):
    if not index in used_pieces:
      stitched_pieces.append(pieces[index])

  return stitched_pieces


###
###  Load relevant information
###

#For deconvolved events
#deco_df = pd.read_hdf(data_path + 'NEW/DE/deco/src_deco/' + file_name, '/DECO/Events')
#deco_df = pd.read_hdf(data_path + 'NEW/Qbb/deco/src/' + file_name, '/DECO/Events')

#For mc events
#deco_df = ldhits(data_path + 'NEW/DE/deco/src_deco/' + file_name)
deco_df = ldhits(data_path + 'NEW/Qbb/deco/src/' + file_name)
deco_df.reset_index(drop=False, inplace=True)
deco_df.rename(columns={'x':'X','y':'Y','z':'Z','energy':'E','event_id':'event'}, inplace=True)

#Correct for known energy calibration issue in NEW MC datasets  Use for double escape src_deco and Qbb
deco_df.E *= 0.9945

try:
  deco_df = deco_df[deco_df.E > 0]
except:
  sys.exit(f'Failed to remove zero energy points for {file_name}.')

if len(deco_df) == 0:
  sys.exit(f'Empty deco file for {file_name}.')


#Initialize container to save output
file_df = pd.DataFrame()
compiled_energy_df = pd.DataFrame()

#Remove events outside of energy ROI of 1% about 2.458 MeV
grps = deco_df.groupby('event').sum()
event_list = grps[(grps.E > 2.458*0.99) & (grps.E < 2.458*1.01)].index.values
#event_list = grps[(grps.E > 1.593*0.99) & (grps.E < 1.593*1.01)].index.values
del grps

#for event_number in [15]:
for event_number in event_list:
  #Initialize event specific information
  piece_df = pd.DataFrame() #Container to save this event output
  event_df = pd.DataFrame() #Container to save info of event

  #Get information about the deconvolved data for this event
  event_df = deco_df[deco_df.event == event_number]
  event_df.reset_index(inplace=True, drop=False)

  #Record the total energy of the event
  event_energy = event_df.E.sum()

  #Single track detection
  #Agglomerative clustering identifies groups sperated by multitrack_distance mm
  #  Retain only the portion of the event containing the most energy.
  pdist = spt.distance.pdist(event_df[['X','Y','Z']].values)
  Z = cls.hierarchy.linkage(pdist)
  cut_tree = cls.hierarchy.cut_tree(Z, height = multitrack_distance)
  event_df = event_df.merge(pd.DataFrame(cut_tree, columns=['Track']), left_index=True, right_index=True)

  #Identify and use only the most energetic track
  track_sums = event_df.groupby('Track').sum()
  energetic_track = track_sums.sort_values('E', ascending=False).iloc[0].name

  event_df = event_df[event_df.Track == energetic_track]
  multitrack_energy = event_df[event_df.Track == energetic_track].E.sum()  #This value is used for single track equivalent
  event_df.drop(columns=['index','Track'], inplace=True)
  event_df.reset_index(drop=True, inplace=True)


  #Small and nearby satellites may be due to deconvolution error
  pdist = spt.distance.pdist(event_df[['X','Y','Z']].values)
  Z = cls.hierarchy.linkage(pdist)
  cut_tree = cls.hierarchy.cut_tree(Z, height = branch_distance)
  event_df = event_df.merge(pd.DataFrame(cut_tree, columns=['Branch']), left_index=True, right_index=True)


  #Identify and use only the most energetic track
  #branch_sums = event_df.groupby('Branch').sum()
  #energetic_branch = branch_sums.sort_values('E', ascending=False).iloc[int(0)].name
  #event_df = event_df[event_df.Branch == energetic_branch]
  #event_df.drop(columns=['Branch'], inplace=True)
  #event_df.reset_index(drop=True, inplace=True)


  kept_branches = []
  branch_extents = []
  branch_Es = []

  if event_df.Branch.nunique() > 1:
    #Keep branches only if they spread across a large span
    # This distinguishes between true satellites and track breaking
    for each_branch in event_df.Branch.unique():
      branch_df = event_df[event_df.Branch == each_branch][['X','Y','Z','E']]
      if len(branch_df) > 1:
        branch_e = branch_df.E.sum()
        branch_Es.append((each_branch, branch_e))
        pdist = spt.distance.pdist(branch_df.values)
        branch_extents.append((each_branch, pdist.max()))

    kept_branches = [each[0] for each in branch_extents if each[1] > branch_extent_param]
    if len(kept_branches) == 0:
      kept_branches = event_df.Branch.unique()

  else:
    kept_branches = [event_df.Branch.unique()[0]]


  if len(kept_branches) == 0:
    kept_branches = [sorted(branch_extents, key=lambda i: i[1], reverse=True)[0][0]]


  #Use only the parts of the event that meet the criteria
  event_df = event_df[event_df.Branch.isin(kept_branches)]
  event_df.reset_index(drop=True, inplace=True)

  if len(event_df) == 0:
    continue

  #Record data about the portion used for track reconstruction
  part_energy = event_df.E.sum()
  min_z = event_df.Z.min()
  max_z = event_df.Z.max()
  max_r = event_df.apply(lambda i: np.sqrt(i.X**2 + i.Y**2), axis=1).max()


  ###
  ###  Use this function if working with deconvolved events
  ###
  #key_points = identify_keypoints(event_df)


  ###
  ###  Add for operating on Monte Carlo true tracks only
  ###
  key_points = [tuple(each) for each in event_df[['X','Y','Z']].values]


  #Place the key_point points in the correct order
  pdist = spt.distance.pdist(key_points)
  Z = cls.hierarchy.linkage(pdist, optimal_ordering=True)
  key_points = [key_points[index] for index in cls.hierarchy.leaves_list(Z)]


  #Recalculate Z using the new order
  pdist = spt.distance.pdist(key_points)
  Z = cls.hierarchy.linkage(pdist, optimal_ordering=True)

  #Identify unusually large gaps in order to break event into pieces
  gaps = [distance(zip(key_points[index-1],key_points[index])) for index in range(len(key_points))][1:]
  gap_cutoff = np.mean(sorted(gaps)[:-1]) + np.std(sorted(gaps)[:-1])
  gap_idx = [index for index, each in enumerate(gaps) if each > gap_cutoff]
  noutliers = len(gap_idx)


  #If the track is segmented, identify the closest point to the end of each
  pieces = []

  if noutliers > 0:
    piece_indexes = [0] + [index+1 for index, each in enumerate(gaps) if each > gap_cutoff] + [len(key_points)]
    for index in range(1,len(piece_indexes)):
      pieces.append(key_points[piece_indexes[index-1]:piece_indexes[index]])

  elif noutliers == 0:
    pieces.append(key_points)

  # If the reconstructed track had large gaps, proceed with branch analysis
  piece_lengths = [len(each) for each in pieces]

  # If there is a piece that only has one point, extend it to include the nearest point to it
  while 1 in piece_lengths:
    #Indexes of each single point piece
    singles_indexes = [index for index, item in enumerate(piece_lengths) if item == 1]

    s_index = singles_indexes[0]

    #Calculate distance between single point and all other key_points
    dists = [spt.distance.cdist(pieces[s_index], pieces[p_index])[0] for p_index in range(len(pieces)) if not p_index == s_index]

    #Determine the minimum distances
    piece_mins = [min(each) for each in dists]
    true_min = min(piece_mins)

    #If the singleton already appears in another piece, remove it from the list
    if true_min == 0:
      pieces.remove(pieces[s_index])

    else:
      #Determine the indexes of those minimums
      piece_index = piece_mins.index(true_min)
      point_index = list(dists[piece_index]).index(true_min)

      #The dists calculation doesn't include the piece index of the singleton
      if piece_index >= s_index:
        piece_index += 1

      #Add in an extra point
      pieces[s_index].append(pieces[piece_index][point_index])

    piece_lengths = [len(each) for each in pieces]


  #If the reconstructed track had large gaps, proceed with branch analysis
  if len(pieces) > 1:

    #Stitch pieces back together one by one until the whole set is reconnected
    all_ends = end_df(pieces, short_version=False, connected=True)
    all_ends = find_end_pairs(all_ends)
    all_ends = find_nodes(all_ends)
    all_ends = groupify(all_ends)

    #Group pieces together until everything is connected, but a limit is set to prevent endless loops
    connection_index=0
    while all_ends.Group.nunique() > 1:
      connection_index += 1
      if connection_index > 30:
        print(f'Excess connections for event {event_number} in {file_name}.')
        break

      try:
        #Find the closest matching of two ends
        diff_groups = ~ (all_ends.Group == all_ends.NP_Group)
        diff_locs = (all_ends.End_Distance > 0)
        restrictions = diff_groups & diff_locs
        best_end = all_ends[restrictions].sort_values('End_Distance').iloc[int(0)]
      except:
#        sys.exit(f'Failed best_end on event {event_number} in {file_name}.')
        print(f'Failed best_end on event {event_number} in {file_name}.')
        break

      if not (best_end.End_Point, best_end.Near_Point) in tuple(tuple(each) for each in pieces):
        #Add a connection between the ends identified as nearest
        pieces.append([best_end.End_Point, best_end.Near_Point])

      all_ends = end_df(pieces, short_version=True, connected=True)
      all_ends = find_end_pairs(all_ends)
      all_ends = find_nodes(all_ends)


      #Divide pieces when a new connection is made to the middle of a piece
      #Record the indexes of locations of new connections
      cut_indexes = []
      for piece_index, each_piece in enumerate(pieces):
        cuts = [0]
        for key_point_index, each_key_point in enumerate(each_piece[1:len(each_piece)-1]):
          if len(all_ends[all_ends.End_Point == each_key_point]) > 0:
            cuts.append(key_point_index+1)
        cuts.append(len(each_piece))
        cut_indexes.append(cuts)

      #Cut pieces using the indexes for cuts recorded in previous section
      complete_pieces = []
      for piece_index, index_list in enumerate(cut_indexes):
        for cut_number, cut_index in enumerate(index_list[:-1]):
          piece_to_cut = pieces[piece_index]
          start_cut = int(cut_index)
          end_cut = (index_list[int(cut_number + 1)]+1)
          new_piece = piece_to_cut[start_cut:end_cut]
          complete_pieces.append(new_piece)

      pieces = complete_pieces

      all_ends = end_df(pieces, short_version=False, connected=True)
      all_ends = find_end_pairs(all_ends)
      all_ends = find_nodes(all_ends)
      all_ends = groupify(all_ends)

    #Join pieces together
    joined_pieces = pieces

    while True:
      if joined_pieces == 0:
        break
      if not len(joined_pieces) == 1:
        joined_pieces = join_pieces(pieces)
      elif len(joined_pieces) == 1:
        break

      if (not joined_pieces == 0) and (not len(pieces) == len(joined_pieces)):
        pieces = joined_pieces
      else:
        if (not joined_pieces == 0) and (len(pieces) == len(joined_pieces)):
          break

    if len(pieces) > 1:
      #Create a new dataframe containing one row for each piece of this event.
      all_ends = end_df(pieces, short_version=True, connected=True)
      all_ends = find_end_pairs(all_ends)
      all_ends = find_nodes(all_ends)

      piece_df = all_ends.groupby('Piece_Index').head(int(1))[['Piece_Index','Piece_Length','Node','Connected_Node']]
      piece_df.reset_index(drop=True, inplace=True)

      piece_series = piece_df.apply(lambda i: pieces[int(i.Piece_Index)], axis=1)
      piece_series.name = 'Key_Points'

      piece_df = piece_df.merge(piece_series, left_index=True, right_index=True)

      piece_df['Event'] = event_number

  if len(pieces) == 1:
    piece_length = sum([distance(zip(pieces[0][index],pieces[0][index-1])) for index in range(len(pieces[0]))][1:])
    piece_df = pd.DataFrame(data=[[event_number, 0, piece_length, 0, 0, pieces[0]]], columns=['Event','Piece_Index','Piece_Length','Node','Connected_Node','Key_Points'])

  full_length = piece_df.Piece_Length.sum()
  file_df = pd.concat([file_df,piece_df])

  energy_df = pd.DataFrame(data=[[event_number, event_energy, multitrack_energy, part_energy, full_length, branch_extents, branch_Es, min_z, max_z, max_r, file_name]], columns=['Event','Full_Energy', 'Multitrack_Energy','Part_Energy','Full_Length','Branch_Extent','Branch_Energy','Min_Z','Max_Z','Max_R','File_Name'])
  compiled_energy_df = pd.concat([compiled_energy_df,energy_df])


data_path = data_path + 'dissertation/NEW/Qbb/mc/recon/'
#data_path = data_path + 'dissertation/NEW/DE/mc/recon/'

#Record info for energies
compiled_energy_df.reset_index(drop=True, inplace=True)
if not os.path.exists(data_path + 'energies/'):
  os.mkdir(data_path + 'energies')

compiled_energy_df.to_hdf(data_path + 'energies/' + file_name, '/Energy')

#Record info for pieces
file_df.reset_index(drop=True, inplace=True)
if not os.path.exists(data_path + 'pieces/'):
  os.mkdir(data_path + 'pieces')

file_df.to_hdf(data_path + 'pieces/' + file_name, '/Pieces')
