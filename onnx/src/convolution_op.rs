use ndarray::*;
use num_traits::Float;
use num_traits::real::Real;
use protobuf::text_format::print_to;
use crate::onnx_structure::tensor_proto::DataLocation::DEFAULT;

pub type DataRepresentation<F> = Array4<F>;

// Padding (specific way of adding zeros to the input matrix) kind used in the convolution.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Padding {
  // explicit padding (specified in "pads" parameter)
  NotSet,
  // output has same shape as input; if odd padding number, extra-padding added at bottom
  SameUpper,
  // output has same shape as input; if odd padding number, extra-padding added at top
  SameLower,
  // no padding
  Valid,
}

// Rust implementation of a convolutional layer.
// The weight matrix (aka kernel) shall have dimension (in that order)
// channels/groups(input channels), feature maps(output channels), kernel width, kernel height,
pub struct ConvolutionLayer<F: Float> {
  pub(in crate) kernel: Array4<F>,
  pub(in crate) bias: Option<Array1<F>>,
  pub(in crate) auto_pad: Padding,
  pub(in crate) dilations: Option<Array2<i32>>,
  pub(in crate) group: Option<i32>,
  pub(in crate) pads: Array1<F>,
  pub(in crate) strides: Array1<F>,
}

impl<F: 'static + Float + std::ops::AddAssign + std::default::Default> ConvolutionLayer<F> where f32: From<F> {
  // Creates new convolution layer.
  pub(crate) fn new(
    kernel: Array4<F>,
    bias: Option<Array1<F>>,
    auto_pad: Padding,
    dilations: Option<Array2<i32>>,
    group: Option<i32>,
    pads: Array1<F>,
    strides: Array1<F>,
  ) -> ConvolutionLayer<F> {
    ConvolutionLayer { kernel, bias, auto_pad, dilations, group, pads, strides }
  }

  /// Creates new convolution layer. The weights are given in ONNX Tensorflow layout:
  /// feature maps(output channels), channels/groups(input channels), kernel height, kernel width
  /// converted into:
  /// channels/groups(input channels), feature maps(output channels), kernel width, kernel height,
  pub fn new_onnx_tensor_flow(
    kernel: Array4<F>,
    bias: Option<Array1<F>>,
    auto_pad: Padding,
    dilations: Option<Array2<i32>>,
    group: Option<i32>,
    pads: Array1<F>,
    strides: Array1<F>,
  ) -> ConvolutionLayer<F> {
    let permuted_view = kernel.view().permuted_axes([1, 0, 3, 2]);
    // Hack to fix the memory layout, permuted axes makes a
    // col major array / non-contiguous array from kernel
    let permuted_array: Array4<F> = Array::from_shape_vec(permuted_view.dim(), permuted_view.iter().copied().collect()).unwrap();
    ConvolutionLayer::new(permuted_array, bias, auto_pad, dilations, group, pads, strides)
  }

  /// Analog to conv2d.
  pub fn convolve(&self, image: &DataRepresentation<F>) -> DataRepresentation<F> {
    conv2d(
      &self.kernel,
      image,
      self.bias.as_ref(),
      self.auto_pad,
      self.dilations.as_ref(),
      self.group,
      &self.pads,
      &self.strides,
    )
  }
}

/// OPSET VERSION: 8
/// Performs a convolution on the given image data using this layers parameters.
/// We always convolve on flattened images and expect the input array in im2col
/// style format.
///
/// Read more here:
/// - <https://leonardoaraujosantos.gitbook.io/artificial-inteligence/machine_learning/deep_learning/convolution_layer/making_faster>
///
/// Input:
///
///  - im2d(batch size, channels, height, width): Array4.
///  - kernel_weights(F=#Filters(channels/groups), C=#ChannelsOut(feature maps), width, height): Array4. (Feature Maps->#output volume)
///  - bias: Array1. (Bias, is added to each channel (after having adding each Hadamard Product))
///  - auto_pad: ["NOTSET"->pads has meant not to be None (manually specified padding),
///               "SAME_UPPER"->padding equally split between axis(if odd number, extra padding added to bottom),
///               "SAME_LOWER"->padding equally split between axis(if odd number, extra padding added to top,
///               "VALID"->no padding
///               ]
///  - dilations: Array1. (Dilation over kernel a.k.a. w filter)
///  - group: i32. Number of groups
///  - kernel_shape: Array1. If not None means the shape of kernel a.k.a. w filter. Since it's not required from the standard, it's inferred from kernel_weights
///  - pads: Array1. Manual padding specified accordingly to auto_pad
///  - strides: Array1. Moving offset over each x input axis.
/// Returns:
/// -----------------------------------------------
/// - out: Output data, of shape (B, F, H', W')
pub fn conv2d<'a, T, V, F: 'static + Float + std::ops::AddAssign + std::default::Default>(
  kernel_weights: T,
  im2d: T,
  bias: Option<&Array1<F>>,
  auto_pad: Padding,
  dilations: Option<&Array2<i32>>,
  group: Option<i32>,
  pads: V, //Option<&Array1<F>>
  strides: V, //Option<&Array1<F>>
) -> DataRepresentation<F>
  where
  // This trait bound ensures that kernel and im2d can be passed as owned array or view.
  // AsArray just ensures that im2d can be converted to an array view via ".into()".
  // Read more here: https://docs.rs/ndarray/0.12.1/ndarray/trait.AsArray.html
    T: AsArray<'a, F, Ix4>,
    V: AsArray<'a, F, Ix1>, f32: From<F>
{
  // Initialisations
  let im2d_arr: ArrayView4<F> = im2d.into();
  let kernel_weights_arr: ArrayView4<F> = kernel_weights.into();
  let strides_arr: ArrayView1<F> = strides.into();
  let pads_arr: ArrayView1<F> = pads.into();
  let im_col: Array2<F>; // output of fn: im2col_ref()
  let new_im_height: usize;
  let new_im_width: usize;
  let weight_shape = kernel_weights_arr.shape();

  assert!(im2d_arr.shape()[1] == (weight_shape[0] * group.unwrap() as usize) && weight_shape[1] % group.unwrap() as usize == 0);

  let mut num_filters = weight_shape[0];
  match group {
    Some(g) => num_filters = num_filters / g as usize,
    None => {}
  }
  let num_channels_out = weight_shape[1];
  let kernel_height = weight_shape[2];
  let kernel_width = weight_shape[3];
  let mut pads_height_start: usize = 0;
  let mut pads_height_end: usize = 0;
  let mut pads_width_start: usize = 0;
  let mut pads_width_end: usize = 0;
  if auto_pad == Padding::NotSet {
    let pads_height_start_as_f = *pads_arr.get(0).unwrap();
    let pads_height_start_as_f32: f32 = pads_height_start_as_f.into();
    pads_height_start = pads_height_start_as_f32 as usize;
    let pads_height_end_as_f = *pads_arr.get(2).unwrap();
    let pads_height_end_as_f32: f32 = pads_height_end_as_f.into();
    pads_height_end = pads_height_end_as_f32 as usize;
    let pads_width_start_as_f = *pads_arr.get(1).unwrap();
    let pads_width_start_as_f32: f32 = pads_width_start_as_f.into();
    pads_width_start = pads_width_start_as_f32 as usize;
    let pads_width_end_as_f = *pads_arr.get(3).unwrap();
    let pads_width_end_as_f32: f32 = pads_width_end_as_f.into();
    pads_width_end = pads_width_end_as_f32 as usize;
  }

  let im_batch_size = im2d_arr.len_of(Axis(0));
  let im_channel = im2d_arr.len_of(Axis(1));
  let im_height = im2d_arr.len_of(Axis(2));
  let im_width = im2d_arr.len_of(Axis(3));
  let im_height_stride_as_f = *strides_arr.get(0).unwrap();
  let im_height_stride_as_f32: f32 = im_height_stride_as_f.into();
  let im_height_stride = im_height_stride_as_f32 as usize;
  let im_width_stride_as_f = *strides_arr.get(1).unwrap();
  let im_width_stride_as_f32: f32 = im_width_stride_as_f.into();
  let im_width_stride = im_width_stride_as_f32 as usize;

  // Calculate output shapes H', W' for two types of Padding
  match auto_pad {
    Padding::SameLower => {
      // H' = (H / stride).ceil()
      // W' = (W / stride).ceil()
      let new_im_height_float = (im_height as f32 / im_height_stride as f32).ceil();
      let new_im_width_float = (im_width as f32 / im_width_stride as f32).ceil();

      new_im_height = new_im_height_float as usize;
      new_im_width = new_im_width_float as usize;
    }
    Padding::SameUpper => {
      // H' = (H / stride).ceil()
      // W' = (W / stride).ceil()
      let new_im_height_float = (im_height as f32 / im_height_stride as f32).ceil();
      let new_im_width_float = (im_width as f32 / im_width_stride as f32).ceil();

      new_im_height = new_im_height_float as usize;
      new_im_width = new_im_width_float as usize;
    }
    Padding::NotSet => {
      // H' = {[H - HH + (2*padding)] / stride}+ 1
      // W' = {[W - WW + (2*padding)] / stride} + 1
      new_im_height = ((im_height - kernel_height + (pads_height_start + pads_height_end)) / im_height_stride) + 1;
      new_im_width = ((im_width - kernel_width + (pads_width_start + pads_width_end)) / im_width_stride) + 1;
    }
    Padding::Valid => {
      // H' =  ((H - HH) / stride_height) + 1
      // W' =  ((W - WW) / stride_width) + 1
      new_im_height = ((im_height - kernel_height) / im_height_stride) + 1;
      new_im_width = ((im_width - kernel_width) / im_width_stride) + 1;
    }
  };

  // weights.reshape(F, HH*WW*C)
  let filter_col = kernel_weights_arr
    .into_shape((num_channels_out, kernel_height * kernel_width * num_filters))
    .unwrap();

  if auto_pad != Padding::Valid {
    let mut pad_num_h = 0;
    let mut pad_num_w = 0;
    let mut pad_top = 0;
    let mut pad_bottom = 0;
    let mut pad_left = 0;
    let mut pad_right = 0;
    if auto_pad == Padding::SameUpper || auto_pad == Padding::SameLower {
      (pad_num_h, pad_num_w, pad_top, pad_bottom, pad_left, pad_right) = get_padding_size(im_height, im_width, im_height_stride, im_width_stride, kernel_height, kernel_width);
    } else if auto_pad == Padding::NotSet {
      pad_top = pads_height_start;
      pad_bottom = pads_height_end;
      pad_left = pads_width_start;
      pad_right = pads_width_end;
      pad_num_h = pads_height_start + pads_height_end;
      pad_num_w = pads_width_start + pads_width_end;
    }
    let mut im2d_arr_pad: Array4<F> = Array4::zeros((
      im_batch_size,
      im_channel,
      im_height + pad_num_h,
      im_width + pad_num_w,
    ));
    let pad_bottom_int = (im_height + pad_num_h) - pad_bottom;
    let pad_right_int = (im_width + pad_num_w) - pad_right;
    // https://github.com/rust-ndarray/ndarray/issues/823
    im2d_arr_pad
      .slice_mut(s![.., .., pad_top..pad_bottom_int, pad_left..pad_right_int])
      .assign(&im2d_arr);

    let im_height_pad = im2d_arr_pad.len_of(Axis(2));
    let im_width_pad = im2d_arr_pad.len_of(Axis(3));

    im_col = im2col_ref(
      im2d_arr_pad.view(),
      kernel_height,
      kernel_width,
      im_height_pad,
      im_width_pad,
      im_channel,
      im_height_stride,
      im_width_stride,
      dilations,
    );
  } else {
    im_col = im2col_ref(
      im2d_arr,
      kernel_height,
      kernel_width,
      im_height,
      im_width,
      im_channel,
      im_height_stride,
      im_width_stride,
      dilations,
    );
  }

  let filter_transpose = filter_col.t();
  let mul = im_col.dot(&filter_transpose);
  let output = mul
    .into_shape((new_im_height, new_im_width, im_batch_size, num_channels_out))
    .unwrap()
    .permuted_axes([2, 3, 0, 1]);

  add_bias(&output, bias)
}

pub(in crate) fn get_padding_size(
  input_h: usize,
  input_w: usize,
  stride_h: usize,
  stride_w: usize,
  kernel_h: usize,
  kernel_w: usize,
) -> (usize, usize, usize, usize, usize, usize) {
  let pad_along_height: usize;
  let pad_along_width: usize;
  let idx_0: usize = 0;

  if input_h % stride_h == idx_0 {
    pad_along_height = (kernel_h - stride_h).max(idx_0);
  } else {
    pad_along_height = (kernel_h - (input_h % stride_h)).max(idx_0);
  };
  if input_w % stride_w == idx_0 {
    pad_along_width = (kernel_w - stride_w).max(idx_0);
  } else {
    pad_along_width = (kernel_w - (input_w % stride_w)).max(idx_0);
  };

  let pad_top = pad_along_height / 2;
  let pad_bottom = pad_along_height - pad_top;
  let pad_left = pad_along_width / 2;
  let pad_right = pad_along_width - pad_left;

  // yes top/bottom and right/left are swapped. No, I don't know
  // why this change makes it conform to the pytorch implementation.
  (
    pad_along_height,
    pad_along_width,
    pad_bottom,
    pad_top,
    pad_right,
    pad_left,
  )
}

pub(in crate) fn im2col_ref<'a, T, F: 'a + Float + std::default::Default>(
  im_arr: T,
  ker_height: usize,
  ker_width: usize,
  im_height: usize,
  im_width: usize,
  im_channel: usize,
  stride_h: usize,
  stride_w: usize,
  dilations: Option<&Array2<i32>>,
) -> Array2<F>
  where
  // Args:
  //   im_arr: image matrix to be translated into columns, (C,H,W)
  //   ker_height: filter height (hh)
  //   ker_width: filter width (ww)
  //   im_height: image height
  //   im_width: image width
  //
  // Returns:
  //   col: (new_h*new_w,hh*ww*C) matrix, each column is a cube that will convolve with a filter
  //         new_h = (H-hh) // stride + 1, new_w = (W-ww) // stride + 1
    T: AsArray<'a, F, Ix4>
{
  let mut cols_img: Array2<F> = Default::default();
  let mut cont = 0_usize;
  match dilations {
    Some(dilations) => {
      let dilation_h = dilations[[0, 0]] as usize;
      let dilation_w = dilations[[0, 1]] as usize;
      if dilation_h > 1 || dilation_w > 1 {
        let im2d_arr: ArrayView4<F> = im_arr.into();
        let new_h = ((im_height - dilation_h * (ker_height - 1) - 1) / stride_h) + 1;
        let new_w = ((im_width - dilation_w * (ker_width - 1) - 1) / stride_w) + 1;
        println!("h:{}, w:{}", new_h, new_w);
        cols_img = Array2::zeros((new_h * new_w, im_channel * ker_height * ker_width));
        for i in 1..new_h + 1 {
          for j in 1..new_w + 1 {
            let h_start = (i - 1) * stride_h;
            let h_end = ((((i - 1) * stride_h + ker_height) - (i - 1) * stride_h) * dilation_h) + (i - 1) * stride_h;
            let w_start = (j - 1) * stride_w;
            let w_end = ((((j - 1) * stride_w + ker_width) - (j - 1) * stride_w) * dilation_w) + (j - 1) * stride_w;
            let patch = im2d_arr.slice(s![
                  ..,
                  ..,
                  h_start..h_end; dilation_h,
                  w_start..w_end; dilation_w
              ]);
            let patchrow_unwrap: Array1<F> = Array::from_iter(patch.map(|a| *a));

            cols_img.row_mut(cont).assign(&patchrow_unwrap);
            cont += 1;
          }
        }
      }else{
        let im2d_arr: ArrayView4<F> = im_arr.into();
        let new_h = ((im_height - ker_height) / stride_h) + 1;
        let new_w = ((im_width - ker_width) / stride_w) + 1;
        cols_img = Array2::zeros((new_h * new_w, im_channel * ker_height * ker_width));

        for i in 1..new_h + 1 {
          for j in 1..new_w + 1 {
            let patch = im2d_arr.slice(s![
                ..,
                ..,
                (i - 1) * stride_h..((i - 1) * stride_h + ker_height),
                (j - 1) * stride_w..((j - 1) * stride_w + ker_width),
            ]);
            let patchrow_unwrap: Array1<F> = Array::from_iter(patch.map(|a| *a));

            cols_img.row_mut(cont).assign(&patchrow_unwrap);
            cont += 1;
          }
        }
      }
    }
    None => {
      let im2d_arr: ArrayView4<F> = im_arr.into();
      let new_h = ((im_height - ker_height) / stride_h) + 1;
      let new_w = ((im_width - ker_width) / stride_w) + 1;
      cols_img = Array2::zeros((new_h * new_w, im_channel * ker_height * ker_width));

      for i in 1..new_h + 1 {
        for j in 1..new_w + 1 {
          let patch = im2d_arr.slice(s![
                ..,
                ..,
                (i - 1) * stride_h..((i - 1) * stride_h + ker_height),
                (j - 1) * stride_w..((j - 1) * stride_w + ker_width),
            ]);
          let patchrow_unwrap: Array1<F> = Array::from_iter(patch.map(|a| *a));

          cols_img.row_mut(cont).assign(&patchrow_unwrap);
          cont += 1;
        }
      }
    }
  };

  cols_img
}

pub(in crate) fn add_bias<F>(x: &Array4<F>, bias: Option<&Array1<F>>) -> Array4<F>
  where
    F: 'static + Float + std::ops::AddAssign,
{
  if let Some(bias_array) = bias {
    assert_eq!(bias_array.shape()[0], x.shape()[1], "Bias array has the wrong shape {:?} for vec of shape {:?}", bias_array.shape(), x.shape());
    // Yes this is really necessary. Broadcasting with ndarray-rust
    // starts at the right side of the shape, so we have to add
    // the axes by hand (else it thinks that it should compare the
    // output width and the bias channels).
    (x + &bias_array
      .clone()
      .insert_axis(Axis(1))
      .insert_axis(Axis(2))
      .broadcast(x.shape())
      .unwrap())
      .into_dimensionality()
      .unwrap()
  } else {
    x.clone()
  }
}
