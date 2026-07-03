/// <reference types="vite/client" />

import type { JSX as ReactJSX } from 'react'

declare global {
	namespace JSX {
		type Element = ReactJSX.Element
		type ElementType = ReactJSX.ElementType
		interface ElementClass extends ReactJSX.ElementClass {}
		interface ElementAttributesProperty extends ReactJSX.ElementAttributesProperty {}
		interface ElementChildrenAttribute extends ReactJSX.ElementChildrenAttribute {}
		interface IntrinsicAttributes extends ReactJSX.IntrinsicAttributes {}
		interface IntrinsicElements extends ReactJSX.IntrinsicElements {}
	}
}
