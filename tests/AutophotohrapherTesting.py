import unittest
from videoToFrames import *

# These files aren't included in the repo, so they will need to be changed
# to paths of files that exist on your system
img_dir = "../images/"
test_vid = img_dir + "tram.mp4"
test_img = img_dir + "rule_of_thirds/rot_left.png"

config = ["brightness"]


class ArgumentTesting(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = create_parser(exit_on_error=False)
        self.good_args = ["-i", test_vid, "-c", "brightness"]

    def test_required_filepath(self):
        with self.assertRaises(SystemExit) as cm:
            self.parser.parse_args(["-i", test_vid])

        self.assertEqual(cm.exception.code, 2)

    def test_verbose_arg(self):
        self.assertTrue(self.parser.parse_args(self.good_args + ["-v"]).verbose)


class RuleOfThirdsTesting(unittest.TestCase):
    def setUp(self) -> None:
        self.rot = Rule_of_Thirds(cv.imread(test_img))

    def test_split_image(self):
        self.assertTrue(len(Rule_of_Thirds(np.zeros((1080, 1920, 3), dtype=np.uint8)).get_image_splits()) == 9)
        self.assertTrue(len(Rule_of_Thirds(np.zeros((720, 1280, 3), dtype=np.uint8)).get_image_splits()) == 9)
        self.assertTrue(len(Rule_of_Thirds(np.zeros((1440, 2560, 3), dtype=np.uint8)).get_image_splits()) == 9)
        self.assertTrue(len(Rule_of_Thirds(np.zeros((768, 1024, 3), dtype=np.uint8)).get_image_splits()) == 9)

    def test_run_algorithm(self):
        pass


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
                self.assertTrue(
                    list(diff[c].items())[0][0] == "LEFT-CENTER_C" and
                    list(diff[c].items())[1][0] == "CENTER_R-BOTTOM"
                )

    def test_rot_left_rainbow(self):
        image = cv.imread(img_dir + "/rule_of_thirds/rot_left_rainbow.png")
        rot = Rule_of_Thirds(image)
        diff = rot.calc_diff(config)

        for c in config:
            with self.subTest(c=c):
                self.assertTrue(list(diff[c].items())[0][0] == "LEFT-CENTER_C")

    def test_rot_rgb(self):
        image = cv.imread(img_dir + "/rule_of_thirds/rot_rgb.png")
        rot = Rule_of_Thirds(image)
        diff = rot.calc_diff(config)

        for c in config:
            with self.subTest(c=c):
                if c == "brightness":
                    self.assertTrue(list(diff[c].items())[0][1] == 0.0)
                else:
                    pass


if __name__ == '__main__':
    unittest.main()
