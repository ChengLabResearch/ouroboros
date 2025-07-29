"""
Module containing shapes of data.
"""
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, asdict, replace, fields, astuple, make_dataclass, Field, InitVar
from functools import cached_property, reduce
import operator
import sys
from typing import Sequence, Union
from typing_extensions import Self

import numpy as np


class DataShape(ABC):
    def to(self, target: type | Self) -> Self:
        try:
            dec = target if isinstance(target, type) else type(target)
            return dec(**asdict(self))
        except TypeError as te:
            te.add_note(f"{sys._getframe(0).f_code.co_name}: Must target another DataShape or Datashape Type.")
            raise te

    def extract(self, target: type | Self) -> Self:
        try:
            dec = target if isinstance(target, type) else type(target)
            source = asdict(self)
            return dec(**{field.name: source[field.name] for field in fields(target)})
        except TypeError as te:
            te.add_note(f"{sys._getframe(0).f_code.co_name}: Must target another Datashape or Datashape Type.")
            raise te
        except KeyError as ke:
            ke.add_note(f"{sys._getframe(0).f_code.co_name}: Target field(s) must exist in source.")
            raise ke

    def cut(self, target: type | Self) -> Self:
        try:
            target_fields = [field.name for field in fields(target)]
            dec = self.gen([field for field in fields(self) if field.name not in target_fields])
            source = asdict(self)
            return dec(**{k: v for k, v in source.items() if k not in target_fields})
        except TypeError as te:
            te.add_note(f"{sys._getframe(0).f_code.co_name}: Must target another Datashape or Datashape Type.")
            raise te
        except IndexError as ie:
            ie.add_note(f"{sys._getframe(0).f_code.co_name}: Must leave at least 1 field.")
            raise ie

    def merge(self, *targets: type | Self, retain=True) -> Self:
        merge_data = [asdict(target) for target in targets]
        if retain:
            merge_data += [asdict(self)]
        else:
            merge_data = [asdict(self)] + merge_data
        new = reduce(operator.or_, merge_data)
        return self.gen(list(new))(**new)

    def is_sub(self, target: type | Self) -> Self:
        return all([field.name in [field.name for field in fields(target)] for field in fields(self)])

    def __mop(self, other: Self, op) -> Self:
        """ Operator function for operators where order does not matter (add, mult). """
        try:
            return replace(self, **{f.name: op(getattr(self, f.name), getattr(other, f.name)) for f in fields(other)})
        except AttributeError:
            return replace(self, **{f.name: op(getattr(self, f.name), getattr(other, f.name)) for f in fields(self)})

    def __op(self, other: Self, op) -> Self:
        """ Operator function for operators where order does matter (sub, div, mod). """
        try:
            return replace(self, **{f.name: op(getattr(self, f.name), getattr(other, f.name)) for f in fields(other)})
        except AttributeError as ae:
            ae.add_note(f"{op} Failed - Fields of other parameter must be in first.")
            raise ae

    def __cp(self, other: Self, cp) -> Self:
        try:
            return all([cp(getattr(self, f.name), getattr(other, f.name)) for f in fields(other)])
        except AttributeError as ae:
            ae.add_note(f"{cp} Failed - Fields of other parameter must be in first.")
            raise ae

    def __add__(self, other: Self) -> Self:
        return self.__mop(other, operator.add)

    def __sub__(self, other: Self) -> Self:
        return self.__op(other, operator.sub)

    def __mul__(self, other: Self) -> Self:
        return self.__mop(other, operator.mul)

    def __floordiv__(self, other: Self) -> Self:
        return self.__op(other, operator.floordiv)

    def __mod__(self, other: Self) -> Self:
        return self.__op(other, operator.mod)

    def __lt__(self, other: Self) -> Self:
        return self.__cp(other, operator.lt)

    def __gt__(self, other: Self) -> Self:
        return self.__cp(other, operator.gt)

    def __le__(self, other: Self) -> Self:
        return self.__cp(other, operator.le)

    def __ge__(self, other: Self) -> Self:
        return self.__cp(other, operator.ge)

    def __neg__(self) -> Self:
        return type(self)(**{f.name: -getattr(self, f.name) for f in fields(self)})

    def __bool__(self) -> Self:
        return any(asdict(self).values())

    @classmethod
    def transpose(self, target: type | Self) -> tuple:
        """ Generates transpose matrix for compatiable with np.transpose and the like."""
        try:
            if len(fields(self)) != len(fields(target)):
                raise ValueError(f"{sys._getframe(0).f_code.co_name}:"
                                 f"Source and Target must have same number of fields.")

            return [[f.name for f in fields(self)].index(key) for key in [f.name for f in fields(target)]]
        except (TypeError, ValueError) as ve:
            ve.add_note(f"{sys._getframe(0).f_code.co_name}:"
                        f"Target must be a datashape or datashape type with the same fields.")
            raise ve

    @classmethod
    def __param_op(cls, op, *shapes: Self) -> Self:
        fields_dict = [[f.name for f in fields(shape)] for shape in shapes]
        try:
            return cls(**{f.name: op([getattr(shapes[i], f.name)
                       for i in range(len(fields_dict)) if f.name in fields_dict[i]])
                       for f in fields(cls)})
        except ValueError as ve:
            ve.add_note(f"{op} Failed - All fields of {cls} must be in at least one argument.")
            raise ve

    @classmethod
    def param_min(cls, *data_shapes: Self) -> Self:
        return cls.__param_op(min, *data_shapes)

    @classmethod
    def param_max(cls, *data_shapes: Self) -> Self:
        return cls.__param_op(max, *data_shapes)

    @staticmethod
    def gen(fields: list[Field] | list[str]) -> Self:
        try:
            names = fields if type(fields[0]) is str else [field.name for field in fields]
            return make_dataclass("_cg_" + "_".join(names), names, bases=(DataShape,))
        except IndexError as ie:
            ie.add_note(f"{sys._getframe(0).f_code.co_name}: Must leave at least 1 field.")
            raise ie

    @classmethod
    def make_with(cls, val: int) -> Self:
        return cls(*([val] * len(fields(cls))))

    @classmethod
    def drange(cls, start: Self | int | Sequence[int],
               stop: Self | int | Sequence[int],
               step: Self | int | Sequence[int]):
        return DataRange(
                start if isinstance(start, cls) else cls(*start) if isinstance(start, Sequence) else cls.make_with(start),     # noqa: E501
                stop if isinstance(stop, cls) else cls(*stop) if isinstance(stop, Sequence) else cls.make_with(stop),
                step if isinstance(step, cls) else cls(*step) if isinstance(step, Sequence) else cls.make_with(step))

    @classmethod
    def range_at_interval(cls, start: Self | int | Sequence[int],
                          stop: Self | int | Sequence[int],
                          interval: Self | int | Sequence[int]):
        istart = start if isinstance(start, cls) else cls(*start) if isinstance(start, Sequence) else cls.make_with(start)     # noqa: E501
        istop = stop if isinstance(stop, cls) else cls(*stop) if isinstance(stop, Sequence) else cls.make_with(stop)
        interval = interval if isinstance(interval, cls) else cls(*interval) if isinstance(interval, Sequence) else cls.make_with(interval)     # noqa:E501
        istep = -((istop - istart) // -interval).extract(interval)
        return DataRange(istart, istop, istep)


@dataclass
class ProjOrder(DataShape): Theta: int; Y: int; X: int      # noqa: E701,E702
@dataclass
class SinoOrder(DataShape): Y: int; Theta: int; X: int      # noqa: E701,E702
@dataclass
class BackwardsSinoOrder(DataShape): X: int; Theta: int; Y: int      # noqa: E701,E702
@dataclass
class ImageStack(DataShape): Z: int; Y: int; X: int      # noqa: E701,E702


@dataclass
class ReconOrder(DataShape):
    Y: int; Z: int; X: int      # noqa: E701,E702

    @classmethod
    def of(cls, source: DataShape):
        try:
            return cls(Y=source.Y, Z=source.X, X=source.X)
        except AttributeError as ae:
            ae.add_note("Source shape should have X and Y dimensions.")
            raise ae


@dataclass
class ImgSlice(DataShape): Y: int; X: int          # noqa: E701,E702
@dataclass
class Proj(DataShape): Y: int; X: int              # noqa: E701,E702
@dataclass
class YSlice(DataShape): Theta: int; X: int     # noqa: E701,E702
@dataclass
class XSlice(DataShape): Theta: int; Y: int     # noqa: E701,E702


@dataclass
class Y(DataShape): Y: int      # noqa: E701,E702
@dataclass
class X(DataShape): X: int      # noqa: E701,E702
@dataclass
class Theta(DataShape): Theta: int      # noqa: E701,E702
@dataclass
class Z(DataShape): Z: int      # noqa: E701,E702


# If we don't actually know the dimensions of a 3D array.
# Avoids X/Y/Z to prevent being used where it shouldn't.
@dataclass
class GenericOrder(DataShape): A: int; B: int; C: int      # noqa: E701,E702


# ????
NPString = namedtuple("NPString", 'T')


@dataclass
class DataRange(object):
    start: DataShape; stop: DataShape; step: DataShape      # noqa: E701,E702

    def __post_init__(self):
        if not (self.step.is_sub(self.stop) and self.stop.is_sub(self.start)):
            raise ValueError("Step and Stop fields must be in Start.")

#         for key, val in asdict(self.step).items():
#             if val == 0:
#                 raise ValueError(f"Field {key} has a 0 step and would not progress.")

    def get_iter(self, conv: 'TFIter' = None):
        self.iter_conv = conv
        fetched_iter = self.__iter__()
        del self.iter_conv
        return fetched_iter

    @cached_property
    def shape(self) -> DataShape:
        """ Shape of the area iterated over. """
        return self.stop - self.start.extract(self.stop).extract(self.step)

    @cached_property
    def length(self) -> DataShape:
        """ Number of steps in each dimension of the iteration. """
        return -(self.shape // -self.step)

    def __len__(self) -> int:
        """ Total number of steps in the iteration. """
        s_len = self.length
        # print(s_len)
        return int(np.ravel_multi_index(astuple(s_len - s_len.make_with(1)), astuple(s_len))) + 1

    def __iter__(self) -> '_DataRangeIter':
        return _DataRangeIter(self.start, self.stop, self.step,
                              None if not hasattr(self, "iter_conv") else self.iter_conv)


@dataclass
class _DataRangeIter(DataRange):
    conv: InitVar[callable] = None

    def __post_init__(self, conv: callable):
        self.__it = np.ndindex(astuple(self.length))             # nditerator
        if conv is not None:
            self.conv = conv(self)
        else:
            self.conv = self.__pos

        # Need to make math quicker.
        # Probably a sign DataShape design is off!
        self.__step_start = astuple(self.start.extract(self.step))
        self.__step = astuple(self.step)
        self.__step_f = [f.name for f in fields(self.step)]

    def __iter__(self):
        return self

    def __pos(self, step_pos):
        return replace(self.start, **{self.__step_f[i]: step_pos[i] for i in range(len(self.__step_f))})

    def __next__(self):
        # Get index of step (next.(self__it)), find the total traversal (operator.mul)
        # and then the position in the step dimension (operator.add)
        step_pos = tuple(map(operator.add, self.__step_start,
                         map(operator.mul, self.__step, next(self.__it))))

        return self.conv(step_pos)


class TFIter(ABC):
    @abstractmethod
    def __call__(self, pos) -> DataShape:
        return pos


class IntIter(TFIter):
    """ TODO: 1d only """
    def __init__(self, dr: DataRange):
        # Increment the first field by 1 to enable clean finishes.
        # increment
        self.__extended_block = astuple(dr.length + dr.start)

    def __call__(self, pos) -> int:
        return np.ravel_multi_index(pos, self.__extended_block)


class ContigMemIter(TFIter):
    def __init__(self, dr: DataRange, offset: int, shape: DataShape, stride: DataShape,
                 jump: list[Union[str, fields]]):
        # increment
        self.offset = offset
        self.stride = stride

        self.__drshape = asdict(dr.shape)
        self.__drstep = asdict(dr.step)
        self.__shape = asdict(shape)

        self.__step_stride = astuple(stride.extract(dr.step))
        self.__step_per_addr, self.__contig_volume = self.__contig_calc(jump)
        self.__counter = 0
        self.__cached_val = None

    @property
    def contig_stride(self):
        return self.__contig_volume

    def __contig_calc(self, break_fields):
        stride_dict = asdict(self.stride)
        stride_keys = list(stride_dict.keys())

        break_fields = [f if not isinstance(f, Field) else f.name for f in break_fields]
        contig_field = None
        step_per_addr = None

        for field in reversed(stride_keys):
            if field in break_fields:
                # Break fields are in separate files (or are otherwise non-contiguous), so anything in the shape
                # before the last of them is non-contiguous.
                break
            if field in self.__drshape:
                if self.__drstep[field] != 1:
                    # Not contiguous from here if it's more than one step per read, so it's this field and end.
                    contig_field = field
                    step_per_addr = 1
                    break
                else:
                    contig_field = field
                    step_per_addr = self.__drshape[field]
                    continue

            # Otherwise, this field is fully contiguous in memory.
            contig_field = field
            step_per_addr = None

        if contig_field is None:
            raise IndexError(f"Break fields must not include last field of shape; {break_fields} in {stride_keys}")

        if step_per_addr is not None:
            return step_per_addr, step_per_addr * stride_dict[contig_field]
        else:
            return 1, self.__shape[contig_field] * stride_dict[contig_field]

    def __call__(self, step_pos) -> int:
        # Break Fields offset issue?
        if self.__counter % self.__step_per_addr == 0:
            self.__cached_val = np.sum(tuple(map(operator.mul, step_pos, self.__step_stride)), dtype=int) + self.offset
        self.__counter += 1
        return self.__cached_val


class MemAddressIter(TFIter):
    def __init__(self, dr: DataRange, offset: int, stride: DataShape):
        # Increment the first field by 1 to enable clean finishes.
        # Do we need this here? don't think so  increment = DataShape.gen([fields(dr.len))[0]])(1)
        # increment
        self.offset = offset
        self.stride = stride
        self.__step_stride = astuple(stride.extract(dr.step))

    def __call__(self, step_pos) -> int:
        # Break Fields offset issue?
        return np.sum(tuple(map(operator.mul, step_pos, self.__step_stride))) + self.offset


class SliceIter(TFIter):
    def __init__(self, dr: DataRange, shape: DataShape):
        self.__base = {field.name: slice(None, None, None) for field in fields(shape)}
        self.__step_f = [f.name for f in fields(dr.step)]

    def __call__(self, pos) -> tuple:
        # print(f"{val}|{self.__base.keys()}")
        return tuple([base if key not in self.__step_f else pos[self.__step_f.index(key)]
                      for key, base in self.__base.items()])


class SliceStepIter(TFIter):
    def __init__(self, dr: DataRange, shape: DataShape):
        self.__shape = asdict(shape)
        self.__step_f = {f.name: i for i, f in enumerate(fields(dr.step))}
        self.__step_v = asdict(dr.step)
        print(self.__step_f)
        print(self.__step_v)

    def __call__(self, pos) -> tuple:
        # print(f"{val}|{self.__base.keys()}")
        pos_step = {field: np.s_[pos[index]: min(pos[index] + self.__step_v[field], self.__shape[field])]
                    for field, index in self.__step_f.items()}

        return tuple([np.s_[:] if key not in self.__step_f else pos_step[key] for key in self.__shape])
