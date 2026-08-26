Had to redirect pip because it ran out of space on the root filesystem (/tmp). So now it installed to | '/home/vincent' | .My code and the venv itself I have in | `/home/vincent/projects/Bremen` |. Video footage (input) I would put in a lyra/cache folder as well (I cant create subfolders like lyra/cache/vincent/subfolders.. so I just store everything in lyra/cache directly)?
What do I do with outputs, since they will probably have to be in lyra cache but I dont want them deleted in between sessions or before I could pull them?
How do I best upload files, currently have a test file in the git project?
Always shows 1,2,7 as occupied/having running tasks but they dont have any running. Only the other GPUs are shown in the final table when checking nvidia-smi.
I am running everything on yolo right now and the first step of the yolo model is detecting people in the frame and setting their bounding box. The detection rate however for the streamline yolo model is horribly low. I know that I should use streamline models to make it reproduceable and not dependent on other factors. Would a pretrained model on a public dataset still count as that? I would maybe use a model as a detector first then. The detection rate without pre training in my first test runs was 0.58 players per frame (there are always at least 20 players in the frame)


OpenPose is the most cited Bottom-Up Model (Keypoint detection first grouping into person second) but has not had any new commits since 2024 and looks discontinued. Is that a problem?
A lot of the GPUs Memories are occupied even though nothing is reserved in the Lyra spreadsheet and no processes are running

YOLO tiled: 451 records, 0.58/frame
OpenPose tiled: 5,989 records, 7.66/frame