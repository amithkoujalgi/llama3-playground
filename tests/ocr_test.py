import cv2
from boxdetect import config
from boxdetect.pipelines import get_checkboxes, get_boxes


# #
# # image = Image.open(file_name)
# # plt.figure()
# # plt.imshow(image)
# # plt.show()
#
# cfg = config.PipelinesConfig()
#
# # important to adjust these values to match the size of boxes on your image
# cfg.width_range = (12, 24)
# cfg.height_range = (12, 24)
#
# # the more scaling factors the more accurate the results, but also it takes more time to processing
# # too small scaling factor may cause false positives
# # too big scaling factor will take a lot of processing time
# cfg.scaling_factors = [0.7]
#
# # w/h ratio range for boxes/rectangles filtering
# cfg.wh_ratio_range = (0.5, 1.5)
#
# # group_size_range starting from 2 will skip all the groups
# # with a single box detected inside (like checkboxes)
# cfg.group_size_range = (0, 1)
#
# # num of iterations when running dilation transformation (to enhance the image)
# cfg.dilation_iterations = 0
#
# # rects, grouping_rects, image, output_image = get_boxes(
# #     file_name,
# #     cfg=cfg,
# #     plot=True
# # )
# #
# # print(f"rects: {rects}")
# # print(f"grouping_rects: {grouping_rects}")
#
# # plt.figure()
# # plt.imshow(output_image)
# # plt.show()
# # output_image.save(f'{file_name}.out.png')
#
# cfg = config.PipelinesConfig()
# cfg.width_range = (15, 30)
# cfg.height_range = (10, 20)
# cfg.wh_ratio_range = (1, 1)
# cfg.dilation_iterations = 0
# checkboxes = get_checkboxes(
#     file_name,
#     cfg=cfg,
#     plot=True,
#     px_threshold=0.2
# )
# img_to_update = cv2.imread(file_name)
# for c in checkboxes:
#     coordinates = c[0]
#     checked = c[1]
#     print(coordinates)
#     option_color = (0, 0, 0)
#     option_font_scale = 0.5
#     option_thickness = 2
#     option_font = cv2.FONT_ITALIC
#     option_yes = '[YES]'
#     option_no = '[NO]'
#     if checked:
#         cv2.putText(img_to_update, option_yes, (coordinates[0] - coordinates[2], coordinates[1] + coordinates[3]),
#                     option_font, option_font_scale,
#                     option_color,
#                     option_thickness)
#     else:
#         cv2.putText(img_to_update, option_no, (coordinates[0] - coordinates[2], coordinates[1] + coordinates[3]),
#                     option_font, option_font_scale,
#                     option_color,
#                     option_thickness)
# cv2.imwrite(f'{file_name}.out.jpg', img_to_update)


# # AUTOCONFIGURE
# cfg = config.PipelinesConfig()
# # The values I'm providing below is a list of box sizes I'm interested in and want to focus on
# # [(h, w), (h, w), ...]
# cfg.autoconfigure([(18, 18), (20, 20), (22, 22)])
#
#
# checkboxes = get_checkboxes(
#     file_name, cfg=cfg, px_threshold=0.1, plot=False, verbose=True)
# print("Output object type: ", type(checkboxes))
# for checkbox in checkboxes:
#     print("Checkbox bounding rectangle (x,y,width,height): ", checkbox[0])
#     print("Result of `contains_pixels` for the checkbox: ", checkbox[1])
#     print("Display the crop out of checkbox:")
#     plt.figure()
#     plt.imshow(checkbox[2])
#     plt.show()
def identify_boxes(file_name):
    cfg = config.PipelinesConfig()
    # cfg.width_range = (15, 30)
    # cfg.height_range = (10, 20)
    # cfg.scaling_factors = [1]
    # cfg.wh_ratio_range = (0.3, 2)

    cfg.width_range = (15, 22)
    cfg.height_range = (15, 22)
    cfg.scaling_factors = [1]
    cfg.wh_ratio_range = (1, 1)
    cfg.group_size_range = (0, 0)
    cfg.dilation_iterations = 0

    rects, grouping_rects, image, output_image = get_boxes(
        file_name,
        cfg=cfg,
        plot=True
    )

    print(f"rects: {rects}")
    print(f"grouping_rects: {grouping_rects}")


def identify_checkboxes(file_name):
    cfg = config.PipelinesConfig()
    cfg.width_range = (15, 30)
    cfg.height_range = (10, 20)
    cfg.wh_ratio_range = (1, 1)
    cfg.dilation_iterations = 0
    checkboxes = get_checkboxes(
        file_name,
        cfg=cfg,
        plot=True,
        px_threshold=0.2
    )
    img_to_update = cv2.imread(file_name)
    for c in checkboxes:
        coordinates = c[0]
        checked = c[1]
        print(coordinates)
        option_color = (0, 0, 0)
        option_font_scale = 0.5
        option_thickness = 2
        option_font = cv2.FONT_ITALIC
        option_yes = '[YES]'
        option_no = '[NO]'
        if checked:
            cv2.putText(img_to_update, option_yes, (coordinates[0] - coordinates[2], coordinates[1] + coordinates[3]),
                        option_font, option_font_scale,
                        option_color,
                        option_thickness)
        else:
            cv2.putText(img_to_update, option_no, (coordinates[0] - coordinates[2], coordinates[1] + coordinates[3]),
                        option_font, option_font_scale,
                        option_color,
                        option_thickness)
    cv2.imwrite(f'{file_name}.out.jpg', img_to_update)


file_path = '/Users/amithkoujalgi/Desktop/3.png'
# identify_checkboxes(file_name=file_path)
identify_boxes(file_name=file_path)
