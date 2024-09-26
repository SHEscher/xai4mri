"""
Collection of utility functions for `xai4mri`.

    Author: Simon M. Hofmann
    Years: 2023
"""

# %% Imports
from __future__ import annotations

import difflib
import gzip
import math
import pickle  # noqa: S403, RUF100
import sys
import warnings
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path, PosixPath
from typing import Any, Callable, ClassVar

import numpy as np

# %% Paths < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class _DisplayablePath:
    """
    Build `_DisplayablePath` class for `tree()`.

    With honourable mention to 'abstrus':
    https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python

    Notes
    -----
        * This uses recursion.
          It will raise a `RecursionError` on really deep folder trees
        * The tree is lazily evaluated.
          It should behave well on really wide folder trees.
          Immediate children of a given folder are not lazily evaluated, though.

    """

    display_filename_prefix_middle = "├──"
    display_filename_prefix_last = "└──"
    display_parent_prefix_middle = "    "
    display_parent_prefix_last = "│   "

    def __init__(self, path, parent_path, is_last):
        """Initialise _DisplayablePath object."""
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    @property
    def display_name(self):
        """Display path name."""
        if self.path.is_dir():
            return self.path.name + "/"
        return self.path.name

    @classmethod
    def make_tree(
        cls,
        root: str | Path,
        parent: str | Path | None = None,
        is_last: bool = False,
        criteria=None,  # noqa: ANN001
    ):
        """Display the tree starting with the given root."""
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        yield displayable_root

        children = sorted([path for path in root.iterdir() if criteria(path)], key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir():
                yield from cls.make_tree(path, parent=displayable_root, is_last=is_last, criteria=criteria)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):  # noqa: ARG003
        return True

    def displayable(self):
        """Provide paths which can be displayed."""
        if self.parent is None:
            return self.display_name

        _filename_prefix = self.display_filename_prefix_last if self.is_last else self.display_filename_prefix_middle

        parts = [f"{_filename_prefix!s} {self.display_name!s}"]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle if parent.is_last else self.display_parent_prefix_last)
            parent = parent.parent

        return "".join(reversed(parts))


def tree(directory: str | Path) -> None:
    """
    Print the directory tree starting at `directory`.

    Use the same way as `shell` command `tree`.

    !!! example "This leads to output such as:"
        ```plaintext
        directory/
        ├── _static/
        │   ├── embedded/
        │   │   ├── deep_file
        │   │   └── very/
        │   │       └── deep/
        │   │           └── folder/
        │   │               └── very_deep_file
        │   └── less_deep_file
        ├── about.rst
        ├── conf.py
        └── index.rst
        ```
    """
    paths = _DisplayablePath.make_tree(Path(directory))
    for path in paths:
        print(path.displayable())


# %% Timer < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def chop_microseconds(delta: timedelta) -> timedelta:
    """
    Chop microseconds from given time delta.

    :param delta: time delta
    :return: time delta without microseconds
    """
    return delta - timedelta(microseconds=delta.microseconds)


def function_timed(dry_funct: Callable[..., Any] | None = None, ms: bool | None = None) -> Callable[..., Any]:
    """
    Time the processing duration of wrapped function.

    !!! example "How to use the `function_timed`"

        The following returns the duration of the function call without micro-seconds:
        ```python
        # Implement a function to be timed
        @function_timed
        def abc():
            return 2 + 2


        # Call the function and get the processing time
        abc()
        ```

        The following returns micro-seconds as well:
        ```python
        @function_timed(ms=True)
        def abcd():
            return 2 + 2
        ```

    :param dry_funct: *Parameter can be ignored*. Results in output without micro-seconds.
    :param ms: If micro-seconds should be printed, set to `True`.
    :return: Wrapped function with processing time.
    """

    def _function_timed(funct):
        @wraps(funct)
        def wrapper(*args, **kwargs):
            """Wrap function to time the processing duration of wrapped function."""
            start_timer = datetime.now()

            # whether to suppress wrapper: use functimer=False in main funct
            w = kwargs.pop("functimer", True)

            output = funct(*args, **kwargs)

            duration = datetime.now() - start_timer

            if w:
                if ms:
                    print(f"\nProcessing time of {funct.__name__}: {duration} [h:m:s:ms]")

                else:
                    print(f"\nProcessing time of {funct.__name__}: {chop_microseconds(duration)} [h:m:s]")

            return output

        return wrapper

    if dry_funct:
        return _function_timed(dry_funct)

    return _function_timed


# %% Normalizer & numerics o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def normalize(
    array: np.ndarray,
    lower_bound: int | float,
    upper_bound: int | float,
    global_min: int | float | None = None,
    global_max: int | float | None = None,
) -> np.ndarray:
    """
    Min-max-scaling: Normalize an input array to lower and upper bounds.

    :param array: Array to be transformed.
    :param lower_bound: Lower bound `a`.
    :param upper_bound: Upper bound `b`.
    :param global_min: Global minimum.
                       If the array is part of a larger tensor, normalize w.r.t. global min and ...
    :param global_max: Global maximum.
                       If the array is part of a larger tensor, normalize w.r.t. ... and global max
                       (i.e., tensor min/max)
    :return: Normalized array.
    """
    if lower_bound >= upper_bound:
        msg = "lower_bound must be < upper_bound"
        raise ValueError(msg)

    array = np.array(array)
    a, b = lower_bound, upper_bound

    if global_min is not None:
        if not np.isclose(global_min, np.nanmin(array)) and global_min > np.nanmin(array):
            # Allow a small tolerance for global_min
            msg = "global_min must be <= np.nanmin(array)"
            raise ValueError(msg)
        mini = global_min
    else:
        mini = np.nanmin(array)

    if global_max is not None:
        if not np.isclose(global_max, np.nanmax(array)) and global_max < np.nanmax(array):
            # Allow a small tolerance for global_max
            msg = "global_max must be >= np.nanmax(array)"
            raise ValueError(msg)
        maxi = global_max
    else:
        maxi = np.nanmax(array)

    return (b - a) * ((array - mini) / (maxi - mini)) + a


# %% Color prints & I/O << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


class Bcolors:
    r"""
    Use for color print-commands in console.

    !!! example "Usage"
        ```python
        print(Bcolors.HEADER + "Warning: No active frommets remain. Continue?" + Bcolors.ENDC)
        print(Bcolors.OKBLUE + "Warning: No active frommets remain. Continue?" + Bcolors.ENDC)
        ```

    ??? note "For more colors:"

        | Name       | Color Code  |
        |------------|-------------|
        | CSELECTED  | \33[7m      |
        | CBLACK     | \33[30m     |
        | CRED       | \33[31m     |
        | CGREEN     | \33[32m     |
        | CYELLOW    | \33[33m     |
        | CBLUE      | \33[34m     |
        | CVIOLET    | \33[35m     |
        | CBEIGE     | \33[36m     |
        | CWHITE     | \33[37m     |
        | CBLACKBG   | \33[40m     |
        | CREDBG     | \33[41m     |
        | CGREENBG   | \33[42m     |
        | CYELLOWBG  | \33[43m     |
        | CBLUEBG    | \33[44m     |
        | CVIOLETBG  | \33[45m     |
        | CBEIGEBG   | \33[46m     |
        | CWHITEBG   | \33[47m     |
        | CGREY      | \33[90m     |
        | CBEIGE2    | \33[96m     |
        | CWHITE2    | \33[97m     |
        | CGREYBG    | \33[100m    |
        | CREDBG2    | \33[101m    |
        | CGREENBG2  | \33[102m    |
        | CYELLOWBG2 | \33[103m    |
        | CBLUEBG2   | \33[104m    |
        | CVIOLETBG2 | \33[105m    |
        | CBEIGEBG2  | \33[106m    |
        | CWHITEBG2  | \33[107m    |

    ???+ example "For preview use:"
        ```python
        for i in (
            [1, 4, 7] + list(range(30, 38)) + list(range(40, 48)) + list(range(90, 98)) + list(range(100, 108))
        ):  # range(107+1)
            print(i, "\33[{}m".format(i) + "ABC & abc" + "\33[0m")
        ```
    """

    HEADERPINK = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    UNDERLINE = "\033[4m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"  # this is necessary in the end to reset to default print

    DICT: ClassVar = {
        "p": HEADERPINK,
        "b": OKBLUE,
        "g": OKGREEN,
        "y": WARNING,
        "r": FAIL,
        "ul": UNDERLINE,
        "bo": BOLD,
    }


def cprint(string: str, col: str | None = None, fm: str | None = None, ts: bool = False) -> None:
    """
    Colorize and format print-out.

    Add leading time-stamp (fs) if required.

    :param string: Print message.
    :param col: Color:'p'(ink), 'b'(lue), 'g'(reen), 'y'(ellow), OR 'r'(ed).
    :param fm: Format: 'ul'(:underline) OR 'bo'(:bold).
    :param ts: Add leading time-stamp.
    """
    if col:
        col = col[0].lower()
        if col not in {"p", "b", "g", "y", "r"}:
            msg = "col must be 'p'(ink), 'b'(lue), 'g'(reen), 'y'(ellow), 'r'(ed)"
            raise ValueError(msg)
        col = Bcolors.DICT[col]

    if fm:
        fm = fm[0:2].lower()
        if fm not in {"ul", "bo"}:
            msg = "fm must be 'ul'(:underline), 'bo'(:bold)"
            raise ValueError(msg)
        fm = Bcolors.DICT[fm]

    if ts:
        pfx = ""  # collecting leading indent or new line
        while string.startswith("\n") or string.startswith("\t"):
            pfx += string[:1]
            string = string[1:]
        string = f"{pfx}{datetime.now():%Y-%m-%d %H:%M:%S} | " + string

    print(f"{col if col else ''}{fm if fm else ''}{string}{Bcolors.ENDC}")


def _true_false_request(func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap print function with true false request."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrap print function with true false request."""
        func(*args, **kwargs)  # should be only a print command
        tof = input("(T)rue or (F)alse: ").lower()
        if tof not in {"true", "false", "t", "f"}:
            msg = "Must be 'T', 't' or 'T/true', or 'F', 'f', 'F/false'"
            raise ValueError(msg)
        return tof in "true"

    return wrapper


@_true_false_request
def ask_true_false(question: str, col: str = "b") -> None:
    """
    Ask user for input for a given `True`-or-`False` question.

    :param question: Question to be asked to the user.
    :param col: Print-color of question ['b'(lue), 'g'(reen), 'y'(ellow), 'r'(ed)]
    :return: Answer to the question.
    """
    cprint(string=question, col=col)


# %% Text << o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def get_string_overlap(s1: str, s2: str) -> str:
    """
    Find the longest overlap between two strings, starting from the left.

    !!! example
        ```python
        get_string_overlap("Hello there Bob", "Hello there Alice")  # "Hello there "
        ```
    :param s1: First string.
    :param s2: Second string.
    :return: Longest overlap between the two strings.
    """
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, _, size = s.find_longest_match(0, len(s1), 0, len(s2))  # _ = pos_b

    return s1[pos_a : pos_a + size]


# %% OS >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def browse_files(initialdir: str | None = None, filetypes: str | None = None) -> str:
    """
    Interactively browse and choose a file from the finder.

    This function is a wrapper around the `tkinter.filedialog.askopenfilename` function
    and uses a GUI to select a file.

    ??? note
        ARGS MUST BE NAMED 'initialdir' and 'filetypes'.

    :param initialdir: Directory, where the search should start
    :param filetypes: What type of file-ending is searched for (suffix, e.g., `*.jpg`)
    :return: Path to the chosen file.
    """
    import tkinter  # noqa: PLC0415, RUF100

    root = tkinter.Tk()
    root.withdraw()

    kwargs = {}
    if initialdir:
        kwargs.update({"initialdir": initialdir})
    if filetypes:
        kwargs.update({"filetypes": [(filetypes + " File", "*." + filetypes.lower())]})

    return tkinter.filedialog.askopenfilename(parent=root, title="Choose the file", **kwargs)


def bytes_to_rep_string(number_of_bytes: int) -> str:
    """
    Convert the number of bytes into representative string.

    The function is used to convert the number of bytes into a human-readable format.

    !!! note "The function rounds the number of bytes to two decimal places."

    !!! example
        ```python
        print(bytes_to_rep_string(1_500_000))  # 1.5 MB
        print(bytes_to_rep_string(1_005_500_000))  # 1.01 GB
        ```

    :param number_of_bytes: Number of bytes.
    :return: Representative string of the given bytes number.
    """
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(number_of_bytes, 10**3)))
    p = math.pow(10**3, i)
    size_ = round(number_of_bytes / p, 2)

    return f"{size_} {size_name[i]}"


def check_storage_size(obj: Any, verbose: bool = True) -> int:
    """
    Return the storage size of a given object in an appropriate unit.

    !!! example
        ```python
        import numpy as np

        a = np.random.rand(500, 500)
        size_in_bytes = check_storage_size(obj=a, verbose=True)  # "Size of given object: 2.0 MB"
        ```

    :param obj: Any object in the workspace.
    :param verbose: Print human-readable size of the object and additional information.
    :return: Object size in bytes.
    """
    if isinstance(obj, np.ndarray):
        size_bytes = obj.nbytes
        message = ""
    else:
        size_bytes = sys.getsizeof(obj)
        message = "Only trustworthy for pure python objects, otherwise returns size of view object."

    if size_bytes == 0:
        if verbose:
            print("Size of given object equals 0 B")
        return 0

    if verbose:
        print(f"Size of given object: {bytes_to_rep_string(number_of_bytes=size_bytes)} {message}")

    return size_bytes


def compute_array_size(
    shape: tuple[int, ...] | list[int, ...], dtype: np.dtype | int | float, verbose: bool = False
) -> int:
    """
    Compute the theoretical size of a NumPy array with the given shape and data type.

    The idea is to compute the size of the array before creating it to avoid potential memory issues.

    :param shape: Shape of the array, e.g., `(n_samples, x, y, z)`.
    :param dtype: Data type of the array elements (e.g., np.float32, np.int64, np.uint8, int, float)
    :param verbose: Print the size of the array in readable format or not.
    :return: Size of the array in bytes.
    """
    # Get the size of each element in bytes
    element_size = np.dtype(dtype).itemsize
    # Compute the total number of elements
    num_elements = np.prod(shape)
    # Compute the total size in bytes
    total_size_in_bytes = num_elements * element_size
    if verbose:
        print(f"Size of {dtype.__name__}-array of shape {shape}: {bytes_to_rep_string(total_size_in_bytes)}")
    return total_size_in_bytes


# %% Save objects externally & load them o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


@function_timed
def _save_obj(
    obj: Any,
    name: str,
    folder: str | PosixPath | Path,
    hp: bool = True,
    as_zip: bool = False,
    save_as: str = "pkl",
):
    """
    Save object as pickle or numpy file.

    :param obj: object to be saved
    :param name: name of pickle/numpy file
    :param folder: target folder
    :param hp: True: highest protocol of pickle
    :param as_zip: True: zip file
    :param save_as: default is pickle, can be "npy" for numpy arrays
    """
    # Remove suffix here, if there is e.g. "*.gz.pkl":
    if name.endswith(".gz"):
        name = name[:-3]
        as_zip = True
    if name.endswith(".pkl") or name.endswith(".npy") or name.endswith(".npz"):
        save_as = "pkl" if name.endswith(".pkl") else "npy"
        name = name[:-4]

    p2save = Path(folder, name)

    # Create parent folder if not available
    parent_dir = p2save.parent
    parent_dir.mkdir(parents=True, exist_ok=True)

    save_as = save_as.lower()
    if save_as == "pkl":
        open_it, suffix = (gzip.open, ".pkl.gz") if as_zip else (open, ".pkl")
        with open_it(p2save + suffix, "wb") as f:
            if hp:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            else:
                p = pickle.Pickler(f)
                p.fast = True
                p.dump(obj)
    elif save_as == "npy":
        if not isinstance(obj, np.ndarray):
            msg = f"Object must be numpy array, but is {type(obj)}!"
            raise TypeError(msg)
        if as_zip:
            np.savez_compressed(file=p2save, arr=obj)
        else:
            np.save(arr=obj, file=p2save)
    else:
        raise ValueError(f"Format save_as='{save_as}' unknown!")


@function_timed
def _load_obj(name: str, folder: str | PosixPath | Path, **kwargs) -> Any:
    """
    Load a pickle or numpy object into workspace.

    :param name: name of the dataset
    :param folder: target folder
    :return: object
    """
    # Check whether also a zipped version is available: "*.pkl.gz"
    possible_fm = [".pkl", ".pkl.gz", ".npy", ".npz"]

    def _raise_name_issue():
        msg = (
            f"'{folder}' contains too many files which could fit name='{name}'.\n"
            f"Specify full name including suffix!"
        )
        raise ValueError(msg)

    # Check all files in the folder which find the name + *suffix
    found_files = [str(pa) for pa in Path(folder).glob(name + "*")]
    n_zipped = 2
    if not any(name.endswith(fm) for fm in possible_fm):
        # No file-format found, check folder for files
        if len(found_files) == 0:
            raise FileNotFoundError(f"In '{folder}' no file with given name='{name}' was found!")

        if len(found_files) == n_zipped:  # len(found_files) == 2
            # There can be a zipped & unzipped version, take the unzipped version if applicable
            file_name_overlap = get_string_overlap(found_files[0], found_files[1])
            if file_name_overlap.endswith(".pkl"):  # .pkl and .pkl.gz found
                name = Path(file_name_overlap).name
            if file_name_overlap.endswith(".np"):  # .npy and .npz found
                name = Path(file_name_overlap).name + "y"  # .npy
            else:  # if the two found files are not of the same file-type
                _raise_name_issue()

        elif len(found_files) > n_zipped:
            _raise_name_issue()

        else:  # len(found_files) == 1
            name = Path(found_files[0]).name  # un-list

    path_to_file = str(Path(folder, name))

    # Load and return
    if path_to_file.endswith(".pkl") or path_to_file.endswith(".pkl.gz"):  # pickle case
        open_it = gzip.open if path_to_file.endswith(".gz") else open
        with open_it(path_to_file, "rb") as f:
            return pickle.load(f, **kwargs)
    else:  # numpy case
        file = np.load(path_to_file, **kwargs)
        if isinstance(file, np.lib.npyio.NpzFile):  # name.endswith(".npz"):
            # If numpy zip (.npz)
            file = file["arr"]
            # This asserts that object was saved this way: np.savez_compressed(file=..., arr=obj), as
            # in _save_obj():
        return file


def _load_obj_from_abs_path(abs_path: str | Path, **kwargs) -> object:
    """Load an object from an absolute path."""
    abs_path = Path(abs_path).absolute()
    return _load_obj(name=abs_path.name, folder=abs_path.parent, **kwargs)


# %% Warnings o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _experimental(func):
    """Warn that function is in an experimental state."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Wrap function to warn that function is in an experimental state."""
        warnings.warn(
            message=f"{func.__name__} is in an experimental state and may change in the future.",
            category=UserWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return wrapper


# %% Compute tests  o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def run_gpu_test(log_device_placement: bool = False) -> bool:
    """
    Test GPU implementation.

    :param log_device_placement: Log device placement.
    :return: GPU available or not.
    """
    import tensorflow as tf  # noqa: PLC0415, RUF100

    n_gpus = len(tf.config.list_physical_devices("GPU"))
    gpu_available = n_gpus > 0
    cprint(string=f"\nNumber of GPU device(s) available: {n_gpus}", col="g" if gpu_available else "r", fm="bo")

    # Run some operations on the GPU/CPU
    device_name = ["/gpu:0", "/cpu:0"] if gpu_available else ["/cpu:0"]
    tf.debugging.set_log_device_placement(log_device_placement)
    for device in device_name:
        for shape in [6000, 12000]:
            cprint(string=f"\nRun operations on device: {device} using tensor with shape: {shape}", col="b", fm="ul")
            with tf.device(device):
                # Create some tensors and perform an operation
                start_time = datetime.now()
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print(f"{c = }")

                # Create some more complex tensors and perform an operation
                random_matrix = tf.compat.v1.random_uniform(shape=(shape, shape), minval=0, maxval=1)
                dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
                sum_operation = tf.reduce_sum(dot_operation)
                print(f"{sum_operation = }")

            print("\n")
            cprint(string=f"Shape: {(shape, shape)} | Device: {device}", col="y")
            cprint(string=f"Time taken: {datetime.now() - start_time}", col="y")
            print("\n" + "*<o>" * 15)

    cprint(string=f"\nGPU available: {gpu_available}", col="g" if gpu_available else "y", fm="bo")
    return gpu_available


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
