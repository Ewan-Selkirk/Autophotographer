# The paths in the test file are set to be run from the root folder
# as opposed to the 'tests' folder.

import unittest
from videoToFrames import *

# These files aren't included in the repo, so they will need to be changed
# to paths of files that exist on your system
img_dir = "images/"
test_vid = img_dir + "tram.mp4"
test_img = img_dir + "rule_of_thirds/rot_left.png"

config = ["brightness", "sharpness", "color"]


class ArgumentTesting(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = create_parser(exit_on_error=False)
        self.good_args = ["-i", test_vid, "-c"] + config

    def test_required_filepath(self):
        # For the program to work, it needs an input file '-i' and a measure to run '-m'.
        # If neither are present then the program should display an error and exit
        for a in [["-i", test_vid], ["-c", "brightness"], []]:
            with self.subTest(a=a):
                with self.assertRaises(SystemExit) as cm:
                    self.parser.parse_args(a)

                self.assertEqual(cm.exception.code, 2)

    def test_wrong_image(self):
        with self.assertRaises(FileNotFoundError):
            args = ["-i", "does_not_exist.png", "-c", "brightness"]
            Autophotographer((self.parser.parse_args(args))).start()

    def test_wrong_config(self):
        with self.assertRaises(SystemExit) as cm:
            self.parser.parse_args(["-i", "test", "-c", "people"])

        self.assertTrue(cm.exception.code, 2)

    def test_verbose_arg(self):
        # Check the verbose flag is being set when the verbose argument is passed
        self.assertTrue(self.parser.parse_args(self.good_args + ["-v"]).verbose)

    def test_name_arg_tokens(self):
        #
        self.assertEqual(parse_name_arg(self.parser.parse_args(self.good_args + ["-n", "%d_%f"])),
                         f"{datetime.now().strftime('%d-%m-%y')}_tram")

    def test_default_name(self):
        self.assertEqual(parse_name_arg(self.parser.parse_args(self.good_args)),
                         f"{datetime.now().strftime('%d-%m-%y_%H-%M-%S')}_tram")

    def test_name_wrong_token(self):
        self.assertTrue(parse_name_arg(self.parser.parse_args(self.good_args + ["-n", "%doesntexist"])),
                        "%doesntexist")
        self.assertTrue(parse_name_arg(self.parser.parse_args(self.good_args + ["-n", "%f%doesntexist"])),
                        "tramdoesntexist")

    def test_name_escape_tokens(self):
        self.assertEqual(parse_name_arg(self.parser.parse_args(self.good_args + ["-n", "%%"])), "%")

    def test_two_colors(self):
        self.assertCountEqual(parse_config_arg(self.parser.parse_args(["-i", "test", "-c", "color", "colour"]).config),
                              {"color"})


class RuleOfThirdsTesting(unittest.TestCase):
    def setUp(self) -> None:
        self.rot = Rule_of_Thirds(cv.imread(test_img))

    def test_split_image(self):
        self.assertTrue(len(Rule_of_Thirds(np.zeros((1080, 1920, 3), dtype=np.uint8)).get_image_splits()) == 9)
        self.assertTrue(len(Rule_of_Thirds(np.zeros((720, 1280, 3), dtype=np.uint8)).get_image_splits()) == 9)
        self.assertTrue(len(Rule_of_Thirds(np.zeros((1440, 2560, 3), dtype=np.uint8)).get_image_splits()) == 9)
        self.assertTrue(len(Rule_of_Thirds(np.zeros((768, 1024, 3), dtype=np.uint8)).get_image_splits()) == 9)

    def test_run_algorithm(self):
        brightness = Rule_of_Thirds(cv.imread(test_img)).run_algorithm("brightness")
        self.assertEqual(len(brightness), 9)
        self.assertTrue(type(brightness[0]) is not np.ndarray and type(brightness[0]) is np.float64 or float)

    def test_run_fake_algorithm(self):
        with self.assertRaises(AttributeError):
            fake = Rule_of_Thirds(cv.imread(test_img)).run_algorithm("cakes")


class ImageCaptureTesting(unittest.TestCase):
    def test_open_single_image(self):
        ic = ImageCapture(test_img)
        self.assertTrue(ic.read()[0])

    def test_open_fake_image(self):
        with self.assertRaises(FileNotFoundError):
            ic = ImageCapture("llamas_with_hats.tiff")
            ic.read()

    def test_open_group_of_images(self):
        ic = ImageCapture(img_dir + "rule_of_thirds/")
        self.assertTrue(ic.read()[0])
        self.assertGreater(ic.get(cv.CAP_PROP_FRAME_COUNT), 1)
        self.assertEqual(ic.get(cv.CAP_PROP_FRAME_COUNT), len(os.listdir(img_dir + "rule_of_thirds/")))

    def test_get_image_frame(self):
        ic = ImageCapture(img_dir + "rule_of_thirds/")
        self.assertIsInstance(ic.read()[1], np.ndarray)


class TestImagesTesting(unittest.TestCase):
    def test_rot_left(self):
        image = cv.imread(test_img)
        rot = Rule_of_Thirds(image)
        diff = rot.calc_diff(config)

        for c in config:
            with self.subTest(c=c):
                self.assertTrue(list(diff[c].items())[0][0] == "LEFT-CENTER_C")
                self.assertTrue(list(diff[c].items())[-1][1] == 0.0)

    def test_rot_left_bottom(self):
        image = cv.imread(img_dir + "/rule_of_thirds/rot_left_bottom.png")
        rot = Rule_of_Thirds(image)
        diff = rot.calc_diff(config)

        for c in config:
            with self.subTest(c=c):
                self.assertCountEqual(list(diff[c].keys())[:2],
                                      ["LEFT-CENTER_C", "CENTER_R-BOTTOM"])

    def test_rot_left_rainbow(self):
        image = cv.imread(img_dir + "/rule_of_thirds/rot_left_rainbow.png")
        rot = Rule_of_Thirds(image)
        diff = rot.calc_diff(config)

        for c in config:
            with self.subTest(c=c):
                if c == "sharpness":
                    self.assertCountEqual(list(diff[c].values()), [0.0, 0.0, 0.0, 0.0])
                else:
                    self.assertTrue(list(diff[c].items())[0][0] == "LEFT-CENTER_C")

    def test_rot_rgb(self):
        image = cv.imread(img_dir + "/rule_of_thirds/rot_rgb.png")
        rot = Rule_of_Thirds(image)
        diff = rot.calc_diff(config)

        for c in config:
            with self.subTest(c=c):
                self.assertTrue(list(diff[c].items())[0][1] == 0.0)


if __name__ == '__main__':
    unittest.main()
