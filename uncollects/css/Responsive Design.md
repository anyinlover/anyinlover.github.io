# Responsive design

Three key principles of responsive design:

1. A mobile-first approach to design
2. The @media at-rule
3. The use of fluid layouts

## Mobile first

A mobile design is about the content.

When designing for mobile touchscreen devices, be sure to make all the key action items large enough to easily tap with a finger.

`<meta name="viewport" content="width=device-width, initial-scale=1">`

1. It tells the browser to use the device width as the assumed width when interpreting the CSS, instead of pretending to be a full-size desktop browser.
2. it uses initial-scale to set the zoom level at 100% when the page loads.

## Media queries

min-width and max-width are the most common media queries, MDN has the [full list](https://developer.mozilla.org/en-US/docs/Web/CSS/@media).

Instead of `media (min-width: 800px)`, now css support `@media (width >= 800px)`

Media queries can be used to detect operating system dark mode.

```css
@media (prefers-color-scheme: dark) {
  :root {
    --theme-font-color: white;
    --theme-background: #222;
  }
}
```

Write mobile styles first, then work up to larger breakpoints.

![responsive css](../../assets/image/responsive_css.png)

## Fluid layouts

Fluid layout (liquid layout) refers to the use of containers that grow and shrink according to the width of the viewport.

In a fluid layout, the main page container typically doesn’t have an explicit width, or it has one defined using a percentage.

Inside the main container(s), any columns are defined using a percentage. A flexbox layout works as well.

## Responsive images

The first thing you should do is always make sure your images are well compressed.

You should ensure they're not any higher resolution than necessary.

The best practice is to create a few copies of an image, each at a different resolution.

For `<img>` tag, we should use `srcset` attribute.
