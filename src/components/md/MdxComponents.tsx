import React from 'react';
import { MDXRemote } from 'next-mdx-remote/rsc';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import rehypePrettyCode from 'rehype-pretty-code';
import 'katex/dist/katex.min.css';

interface Props {
  source: string;
}

export function MdxRenderer({ source }: Props) {
  return (
    <div className="prose prose-neutral dark:prose-invert max-w-none prose-pre:p-0">
      <MDXRemote
        source={source}
        options={{
          mdxOptions: {
            remarkPlugins: [remarkMath, remarkGfm],
            rehypePlugins: [
              rehypeKatex,
              [
                rehypePrettyCode,
                {
                  theme: 'github-dark',
                  keepBackground: true,
                },
              ],
            ],
          },
        }}
      />
    </div>
  );
}
