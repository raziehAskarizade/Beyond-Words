# ImageToKG

## Objectives
- Convert images to object and 'shape and color' knowledge graph.
- Analyse image's shapes and colors based on their generated knowledge graph using DNNs

## Tasks
- Generate knowledge graph
  - create a shape and color theory knowledge graph 
    - Specify details of the shape and color theory knowledge graph creation task
    - Read EPUB books into strings
      - http://www.dgp.toronto.edu/~donovan/color/
      - Art of Color: Johannes Itten
      - 
    - Write a text to knowledge graph NN model
    - Alternative way is to pay to create over a book or two!
    - 
  - Write a NN model to remove shadow and glow lights from images
  - Write an attribute detector in images based on vocabularies
  - Write a color segmentation NN for images
  - Write a text to knowledge graph NN
  - Find a way to convert color-segments and attributes in images to knowledge graph and relate them to colors knowledge graph
    - Use image segmentation for attributes and for color regions
    - assign each color to overlapping attributes
    - assigning pixel information of each color to each color region
      - neighboring colors
      - percentage of image
  - Add generated knowledge graph to the whole knowledge graph (in the knowledge base)
- Reasoning base on knowledge graph
  - Use pretrained image to kg model to generate knowledge graph 
  - Write a GNN model for knowledge-graphs to score colors in the image with the use of color knowledge graphs database
    - !?

