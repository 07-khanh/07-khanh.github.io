import Link from 'next/link';
import { getPosts } from '@/lib/content';
import { formatDate } from '@/lib/site';
import { pageMetadata } from '@/lib/metadata';

export const metadata = pageMetadata('Writing', 'Explanations and reflections on machine learning, mathematics, and building things.', '/blog');

export default function BlogList() {
  const posts = getPosts();
  return (
    <>
      <header className="page-header">
        <p className="eyebrow">Ideas, worked through</p>
        <h1>Writing</h1>
        <p>Explanations and reflections on machine learning, mathematics, and building things. Writing helps me make sense of what I’m learning.</p>
      </header>
      <div className="entry-list">{posts.map(post => (
        <article className="writing-entry" key={post.slug}>
          <time dateTime={post.date}>{formatDate(post.date)}</time>
          <div><h2 className="font-serif text-2xl font-medium"><Link href={`/blog/${post.slug}`}>{post.title}</Link></h2><p>{post.description}</p></div>
          <span className="entry-arrow" aria-hidden="true">↗</span>
        </article>
      ))}</div>
    </>
  );
}
