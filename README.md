# Usage of Self-similar Features in Flat Objects Recognition

In the context of the image analysis may be interesting to analyze
how many templates of a certain type are present in a given image, for
example to recognize how many medicines of the same type are on the
shelf of a pharmacy or to identify how many advertisements of a given
sponsor are present behind a football interview. There are already several
studies on that and there are also some algorithms that use homography
transformations to identify these images.

Our first aim is to improve the existing works by trying to refine more and more the homographies used.
However the main innovation brought by our work is to try to use some
peculiarities of the image to make improvements in this direction and in
particular using the recurrences in the template, such as the presence of
many identical letters or a background with some repeated patterns; we
call this type of features present in the template as self-similar features.
A lot of these features are discarded by the distance ratio test present in
the most of these algorithms, so we want to find and reintroduce them
since they could be useful to find a more accurate homography.
