import { getPostBySlug, getPosts } from '@/lib/content';
import { markdownToHtml } from '@/lib/markdown';
import { notFound } from 'next/navigation';
import Link from 'next/link';

export async function generateStaticParams() {
  return getPosts().map(p => ({ slug: p.slug }));
}

export default async function BlogPost({ params }: { params: { slug: string } }) {
  const post = getPostBySlug(params.slug);
  if (!post) notFound();
  
  const html = await markdownToHtml(post.content);
  
  return (
    <article className="prose max-w-none font-serif text-[1.125rem]">
      <nav className="not-prose mb-8">
        <Link href="/blog" className="text-sm font-sans text-amber-600 hover:underline">&larr; Back to Blog</Link>
      </nav>
      <header className="mb-12 not-prose border-b border-gray-200 pb-8">
        <time className="text-sm font-sans text-gray-400 font-medium tracking-wide uppercase">{post.meta.date}</time>
        <h1 className="text-4xl font-sans font-bold text-gray-900 mt-4 leading-tight">{post.meta.title}</h1>
      </header>
      <div dangerouslySetInnerHTML={{ __html: html }} />
    </article>
  );
}
