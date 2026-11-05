import { notFound } from 'next/navigation';
import { Article } from '@/components/Article';
import { getPostBySlug, getPosts } from '@/lib/content';
import { pageMetadata } from '@/lib/metadata';

type Props = { params: { slug: string } };

export function generateStaticParams() {
  return getPosts().map(post => ({ slug: post.slug }));
}

export function generateMetadata({ params }: Props) {
  const post = getPostBySlug(params.slug);
  if (!post) notFound();
  return pageMetadata(post.meta.title, post.meta.description, `/blog/${params.slug}`, post.meta.date);
}

export default async function BlogPost({ params }: Props) {
  const post = getPostBySlug(params.slug);
  if (!post) notFound();
  const posts = getPosts();
  const index = posts.findIndex(item => item.slug === params.slug);
  const link = (item: typeof posts[number] | undefined) => item ? { href: `/blog/${item.slug}`, title: item.title } : undefined;
  return <Article {...post.meta} content={post.content} breadcrumbs={[{ href: '/blog', title: 'Writing' }]}
    previous={link(posts[index + 1])} next={link(posts[index - 1])} />;
}
